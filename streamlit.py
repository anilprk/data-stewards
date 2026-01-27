import json
import time
import os
import requests
import pandas as pd
import streamlit as st
from snowflake.core import Root
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col
from urllib.parse import urlparse
from typing import List, Optional
from pydantic import BaseModel, Field
from perplexity import Perplexity


# --- HELPER FUNCTIONS FOR SET AS PRIMARY (BYPASS FLOW FIX) ---
def check_affiliation_exists(session, hcp_npi: str, hco_id: str) -> bool:
    """
    Check if an affiliation record already exists in HCP_HCO_AFFILIATION table.
    """
    if not hcp_npi or not hco_id:
        return False
    
    try:
        query = f"""
            SELECT COUNT(*) as CNT 
            FROM HCP_HCO_AFFILIATION 
            WHERE HCP_NPI = '{hcp_npi}' AND HCO_ID = '{hco_id}'
        """
        result = session.sql(query).collect()
        return result[0].CNT > 0 if result else False
    except Exception as e:
        st.warning(f"Error checking affiliation: {e}")
        return False


def insert_affiliation_record(session, hcp_id:str, hcp_npi: str, hco_data: dict, generate_new_id: bool = False):
    """
    Insert a new affiliation record into HCP_HCO_AFFILIATION table.
    
    Args:
        session: Snowflake session
        hcp_npi: HCP NPI value
        hco_data: Dictionary containing HCO data
        generate_new_id: If True, generate a new HCO_ID for AI-generated records
        
    Returns:
        If generate_new_id is True: Returns the new HCO_ID (int) on success, None on failure
        If generate_new_id is False: Returns True on success, False on failure
    """
    try:
        hco_id = hco_data.get('HCO ID', hco_data.get('HCO_ID', ''))
        
        # For AI-generated records, generate a new HCO_ID
        if generate_new_id or str(hco_id).startswith('ai_generated_') or not hco_id:
            max_id_result = session.sql("SELECT COALESCE(MAX(HCO_ID), 0) AS MAX_ID FROM HCP_HCO_AFFILIATION").collect()
            max_id = max_id_result[0].MAX_ID if max_id_result[0].MAX_ID else 0
            hco_id = int(max_id) + 1
        
        hco_name = hco_data.get('HCO NAME', hco_data.get('HCO_Name', ''))
        hco_address1 = hco_data.get('HCO ADDRESS', hco_data.get('HCO_Address1', ''))
        hco_city = hco_data.get('HCO CITY', hco_data.get('HCO_City', ''))
        hco_state = hco_data.get('HCO STATE', hco_data.get('HCO_State', ''))
        hco_zip = hco_data.get('HCO ZIP', hco_data.get('HCO_ZIP', ''))
        
        # Clean values - escape single quotes
        def clean_val(val):
            if val is None:
                return ''
            return str(val).replace("'", "''")
        
        insert_sql = f"""
            INSERT INTO HCP_HCO_AFFILIATION (HCP_ACCT_ID, HCP_NPI, HCO_ID, HCO_NAME, HCO_ADDRESS1, HCO_CITY, HCO_STATE, HCO_ZIP)
            VALUES (
                '{clean_val(hcp_id)}',
                '{clean_val(hcp_npi)}',
                {hco_id},
                '{clean_val(hco_name)}',
                '{clean_val(hco_address1)}',
                '{clean_val(hco_city)}',
                '{clean_val(hco_state)}',
                '{clean_val(hco_zip)}'
            )
        """
        session.sql(insert_sql).collect()
        
        if generate_new_id or str(hco_data.get('HCO ID', hco_data.get('HCO_ID', ''))).startswith('ai_generated_'):
            return hco_id  # Return the new ID for AI-generated records
        return True
    except Exception as e:
        st.error(f"Error inserting affiliation record: {e}")
        return None if generate_new_id else False


# --- SNOWFLAKE CONNECTION FOR STREAMLIT COMMUNITY CLOUD ---
@st.cache_resource
def get_snowflake_session():
    """
    Creates a Snowflake session using credentials from Streamlit secrets.
    For Streamlit Community Cloud deployment, add secrets in the app settings.
    """
    connection_parameters = {
        "account": st.secrets["snowflake"]["account"],
        "user": st.secrets["snowflake"]["user"],
        "password": st.secrets["snowflake"]["password"],
        "warehouse": st.secrets["snowflake"]["warehouse"],
        "database": st.secrets["snowflake"]["database"],
        "schema": st.secrets["snowflake"]["schema"],
        "role": st.secrets["snowflake"]["role"],
    }
    return Session.builder.configs(connection_parameters).create()


def get_affiliation_priorities_from_llm(session, selected_hcp_data: dict, affiliations: list) -> dict:
    """
    Calls Snowflake Cortex REST API with structured JSON output to rank HCO affiliations by priority.
    
    Returns a dict mapping affiliation key to {"priority": int, "reason": str}
    """
    if not affiliations:
        return {}
    
    # Build the prompt for the LLM
    selected_info = f"""
Selected Healthcare Provider:
- Name: {selected_hcp_data.get('Name', 'N/A')}
- Address: {selected_hcp_data.get('Address Line1', '')} {selected_hcp_data.get('Address Line2', '')}
- City: {selected_hcp_data.get('City', 'N/A')}
- State: {selected_hcp_data.get('State', 'N/A')}
- ZIP: {selected_hcp_data.get('ZIP', 'N/A')}
"""
    
    affiliations_info = "Affiliations to rank:\n"
    for idx, (key, aff) in enumerate(affiliations):
        affiliations_info += f"""
Affiliation {idx + 1} (Key: {key}):
- HCO Name: {aff.get('HCO NAME', 'N/A')}
- HCO Address: {aff.get('HCO ADDRESS', 'N/A')}
- HCO City: {aff.get('HCO CITY', 'N/A')}
- HCO State: {aff.get('HCO STATE', 'N/A')}
- HCO ZIP: {aff.get('HCO ZIP', 'N/A')}
- Source: {aff.get('SOURCE', 'N/A')}
"""
    
    prompt = f"""You are a healthcare data analyst. Analyze the following selected healthcare provider and its potential affiliations. 
Rank each affiliation by priority (1 being highest priority/best match) based on:
1. Geographic proximity (same city, state, ZIP code area)
2. Name similarity or relationship (parent organization, same health system)
3. Address proximity

{selected_info}

{affiliations_info}

Return your response as a valid JSON object with this exact structure:
{{
    "rankings": [
        {{"key": "affiliation_key", "priority": 1, "reason": "Brief explanation of why this is priority 1"}},
        {{"key": "affiliation_key", "priority": 2, "reason": "Brief explanation of why this is priority 2"}}
    ]
}}

Only return the JSON object, no other text. Use the exact keys provided for each affiliation."""

    try:
        # Use Snowflake Cortex REST API with structured JSON output
        account = st.secrets["snowflake"]["account"]
        account_url = account.replace("_", "-").replace(".", "-")
        api_url = f"https://{account_url}.snowflakecomputing.com/api/v2/cortex/inference:complete"
        
        # Get token from session
        token = session.connection.rest.token
        
        headers = {
            "Authorization": f"Snowflake Token=\"{token}\"",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        # Define JSON schema for structured output
        json_schema = {
            "type": "object",
            "properties": {
                "rankings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "priority": {"type": "number"},
                            "reason": {"type": "string"}
                        },
                        "required": ["key", "priority", "reason"]
                    }
                }
            },
            "required": ["rankings"]
        }
        
        request_body = {
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "response_format": {
                "type": "json",
                "schema": json_schema
            }
        }
        
        resp = requests.post(api_url, headers=headers, json=request_body, timeout=60)
        
        if resp.status_code >= 400:
            raise Exception(f"API request failed with status {resp.status_code}: {resp.text}")
        
        # Parse streaming response - collect all content
        response_text = ""
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        response_text += content
                except json.JSONDecodeError:
                    continue
        
        # Parse the structured JSON response
        result = json.loads(response_text.strip())
        
        # Convert to a dictionary keyed by affiliation key
        priority_map = {}
        for ranking in result.get("rankings", []):
            priority_map[str(ranking["key"])] = {
                "priority": ranking["priority"],
                "reason": ranking["reason"]
            }
        return priority_map
        
    except Exception as e:
        st.warning(f"Could not get LLM priority ranking: {e}")
        # Return default priorities if LLM fails
        return {str(key): {"priority": idx + 1, "reason": "Default ordering (LLM unavailable)"} 
                for idx, (key, _) in enumerate(affiliations)}


# Set page to wide layout
st.set_page_config(layout="wide", page_title="HCP Data Steward")

# --- POPUP FUNCTIONS ---
def show_popup_without_button(popup_placeholder, message_type, record_info):
    """
    Renders a custom popup message that auto-dismisses after a delay.
    """
    with popup_placeholder.container():
        st.markdown("""
            <style>
                .st-popup-container {
                    position: fixed;
                    top: 20%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    z-index: 9999;
                    padding: 2rem;
                    border-radius: 10px;
                    background-color: #ffffff;
                    color: #000000;
                    border: 2px solid #4CAF50;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                    text-align: center;
                    min-width: 350px;
                }
                .st-popup-container h4 {
                    color: #4CAF50;
                    font-size: 1.5rem;
                    margin-bottom: 0.5rem;
                }
                .st-popup-container p {
                    font-size: 1rem;
                }
            </style>
            """, unsafe_allow_html=True)
        
        if message_type == "update_success" and 'message' in record_info:
            title = "Update Successful! ✅"
            message = record_info['message']
        elif message_type == "insert_success" and 'message' in record_info:
            title = "Insert Successful! ✅"
            message = record_info['message']
        elif message_type == "primary_success":
            title = "Primary Affiliation Updated! ✅"
            message = f"Primary Parent is set with HCO ID: {record_info.get('hco_id')}."
        else:
            title = "Success! ✅"
            message = "Operation completed successfully."
            
        st.markdown(
            f"""
            <div class="st-popup-container">
                <h4>{title}</h4>
                <p>{message}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    time.sleep(2)
    st.session_state.show_popup = False
    st.session_state.popup_message_info = None
    st.rerun()


# --- NEW, SELF-CONTAINED RAG CHATBOT FUNCTION ---
def render_rag_chatbot(session):
    """
    This function encapsulates all the logic and UI for the
    'Chat with your Documents' RAG chatbot.
    """
    # RAG App Constants
    NUM_CHUNKS = 3
    CORTEX_SEARCH_DATABASE = "CORTEX_ANALYST_HCK"
    CORTEX_SEARCH_SCHEMA = "PUBLIC"
    CORTEX_SEARCH_SERVICE = "CC_SEARCH_SERVICE_CS"
    COLUMNS = ["chunk", "chunk_index", "relative_path", "category"]
    try:
        root = Root(session)
        svc = (
            root.databases[CORTEX_SEARCH_DATABASE]
            .schemas[CORTEX_SEARCH_SCHEMA]
            .cortex_search_services[CORTEX_SEARCH_SERVICE]
        )
    except Exception as e:
        st.error(f"Could not connect to the Cortex Search Service. Error: {e}")
        st.stop()

    # Helper Functions nested inside
    def config_options():
        st.sidebar.title("Chatbot Configuration")
        st.sidebar.selectbox(
            "Select your model:",
            ("mistral-large", "llama3-70b", "llama3-8b", "snowflake-arctic"),
            key="model_name",
        )
        categories = session.sql(
            "select category from docs_chunks_table group by category"
        ).collect()
        cat_list = ["ALL"]
        for cat in categories:
            cat_list.append(cat.CATEGORY)
        st.sidebar.selectbox("Filter by product type:", cat_list, key="category_value")
        st.session_state.rag = st.sidebar.toggle(
            "Use documents as context?", value=True
        )

    def get_similar_chunks_search_service(query):
        if st.session_state.category_value == "ALL":
            response = svc.search(query, COLUMNS, limit=NUM_CHUNKS)
        else:
            filter_obj = {"@eq": {"category": st.session_state.category_value}}
            response = svc.search(query, COLUMNS, filter=filter_obj, limit=NUM_CHUNKS)
        return response.json()

    def create_prompt(myquestion):
        if st.session_state.rag:
            retrieved_chunks_dict = get_similar_chunks_search_service(myquestion)
            context_for_prompt = json.dumps(retrieved_chunks_dict)
            prompt = f"""
                         You are an expert chat assistant that extracts information from the CONTEXT provided
                         between <context> and </context> tags. Be concise and do not hallucinate.
                         If you don´t have the information just say so.
                         <context>{context_for_prompt}</context>
                         <question>{myquestion}</question>
                         Answer:
                         """
            relative_paths = set(
                item["relative_path"]
                for item in retrieved_chunks_dict.get("results", [])
            )
        else:
            prompt = f"Question: {myquestion} Answer:"
            relative_paths = "None"
        return prompt, relative_paths

    def complete(myquestion):
        prompt, relative_paths = create_prompt(myquestion)
        full_cmd = f"SELECT snowflake.cortex.complete('{st.session_state.model_name}', $${prompt}$$) as response"
        df_response = session.sql(full_cmd).collect()
        return df_response, relative_paths

    # RAG UI
    st.subheader("Chat with your Documents")
    docs_available = session.sql("ls @docs").collect()
    list_docs = [doc["name"] for doc in docs_available]
    st.dataframe(list_docs, use_container_width=True)
    config_options()
    question = st.text_input(
        "Ask a question about the documents:",
        placeholder="e.g., What is the provider's primary specialty?",
        label_visibility="collapsed",
    )

    if question:
        with st.spinner("Generating answer..."):
            response, relative_paths = complete(question)
            res_text = response[0].RESPONSE
            st.markdown(res_text)
            if relative_paths != "None":
                with st.expander("Related Documents"):
                    for path in relative_paths:
                        cmd2 = f"select GET_PRESIGNED_URL(@docs, '{path}', 360) as URL_LINK from directory(@docs)"
                        df_url_link = session.sql(cmd2).to_pandas()
                        url_link = df_url_link.iloc[0]["URL_LINK"]
                        display_url = f"Doc: [{path}]({url_link})"
                        st.markdown(display_url)


# --- ENRICHMENT & COMPARISON PAGE FUNCTION ---
def render_enrichment_page(session, selected_hcp_df):
    # --- BACK BUTTON LOGIC ---
    _, btn_col = st.columns([4, 1])
    with btn_col:
        if st.button("⬅️ Back to Search Results"):
            st.session_state.current_view = "main"
            st.session_state.selected_hcp_id = None
            st.rerun()

    # --- Custom CSS for the "Web Report" Look ---
    st.markdown(
        """
    <style>
        div[data-testid="stHorizontalBlock"]:has(div.cell-content),
        div[data-testid="stHorizontalBlock"]:has(div.hco-cell) { border-bottom: 1px solid #e6e6e6; }
        div[data-testid="stHorizontalBlock"]:has(div.cell-content):hover,
        div[data-testid="stHorizontalBlock"]:has(div.hco-cell):hover { background-color: #f8f9fa; }
        .cell-content, .hco-cell { padding: 0.3rem 0.5rem; font-size: 14px; display: flex; align-items: center; height: 48px; }
        .report-header, .hco-header { font-weight: bold; color: #4f4f4f; padding: 0.5rem; }
        .hco-header { border-bottom: 2px solid #ccc; }
        .report-proposed-column { border-left: 2px solid #D3D3D3; padding-left: 1.5rem; }
        .checkbox-container { width: 100%; text-align: center; }
        .checkbox-container div[data-testid="stCheckbox"] { padding-top: 12px; }
        div[data-testid="stExpander"] button { margin-top: -1rem; }
        .hco-cell div[data-testid="stButton"] button { padding: 0.2rem 0.5rem; font-size: 12px; height: 30px; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # --- API Data Enrichment Function ---
    @st.cache_data(ttl=600)
    def get_enriched_data_from_llm(_session, hcp_df, search_query=None):
        if hcp_df.empty:
            return pd.DataFrame()
    
        selected_record = hcp_df.iloc[0].to_dict()

        try:
            api_response = get_consolidated_data_for_hcp(selected_record, model_name="sonar-pro", use_pro_search=True, search_query=search_query)
            return api_response
        
        except Exception as e:
            st.error(f"An error occurred during the AI enrichment process: {e}")
            return pd.DataFrame()

    # --- Main Application Logic for Enrichment Page ---
    #st.title("📑 Current vs. Proposed Comparison Report")
    st.markdown("<h3>📑 Current vs. Proposed Comparison Report</h3>", unsafe_allow_html=True)
    

    if selected_hcp_df.empty:
        st.warning("No HCP data was provided for enrichment.")
        st.stop()
        
    selected_record = selected_hcp_df.iloc[0]
    current_data_dict = { 'ID': selected_record.get('ID', ''), 'Name': selected_record.get('NAME', ''), 'NPI': selected_record.get('NPI', ''), 'Address Line1': selected_record.get('ADDRESS1', ''), 'Address Line2': selected_record.get('ADDRESS2', ''), 'City': selected_record.get('CITY', ''), 'State': selected_record.get('STATE', ''), 'ZIP': selected_record.get('ZIP', '') }
    current_df = pd.DataFrame([current_data_dict])

    # Placeholder for a potential dialog to display over the main content
    dialog_placeholder = st.empty()
    
    # Render reason popup as a modal overlay if the state is set
    if st.session_state.get('show_reason_popup'):
        popup_data = st.session_state.get('reason_popup_data', {})
        
        # Modal overlay styling - use f-string with pre-extracted values
        hco_name = popup_data.get('hco_name', 'Unknown')
        priority = popup_data.get('priority', '-')
        reason = popup_data.get('reason', 'No reason available')
        
        # Use Streamlit's dialog decorator if available (Streamlit 1.33+)
        @st.dialog("🎯 Priority Reasoning")
        def show_reason_dialog():
            st.markdown(f"**Organization:** {hco_name}")
            st.markdown(f"<span style='display: inline-block; background-color: #4CAF50; color: white; padding: 0.25rem 0.75rem; border-radius: 15px; font-weight: bold;'>Priority {priority}</span>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; border-left: 4px solid #1f77b4;'>
                <strong>Reason:</strong><br>{reason}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            if st.button("Close", key="close_dialog_btn", use_container_width=True):
                st.session_state.show_reason_popup = False
                st.session_state.reason_popup_data = None
                st.rerun()
        
        show_reason_dialog()
    
    # Render confirmation dialog if the state is set
    if st.session_state.get('show_confirm_dialog'):
        is_new_record = st.session_state.selected_hcp_id == 'empty_record' or str(selected_record.get('ID')) in ['', 'N/A']
        with dialog_placeholder.container():
            action_text = "insert a new record" if is_new_record else "update the selected fields"
            st.warning(f"Are you sure you want to {action_text}? This action cannot be undone.", icon="⚠️")
            
            # --- MODIFIED: Display selected changes and remaining records in two tables ---
            approved_df_cols = st.session_state.get('approved_cols', [])
            proposed_record = st.session_state.get('proposed_record', {})
            
            provider_mapping = { 
                "Name": "NAME", "Address Line1": "ADDRESS1", "Address Line2": "ADDRESS2",
                "City": "CITY", "State": "STATE", "ZIP": "ZIP" 
            }
            
            # Table 1: Changes to be applied
            changes_to_display = []
            for field_label, db_col in provider_mapping.items():
                if field_label in approved_df_cols:
                    current_val = selected_record.get(db_col)
                    proposed_val = proposed_record.get(field_label)
                    changes_to_display.append([field_label, current_val, proposed_val])
            
            if changes_to_display:
                st.markdown("---")
                st.markdown(f"**Changes to be applied for Account ID: `{selected_record.get('ID')}`**")
                
                # Use st.columns to simulate a two-column table for styling
                cols_header = st.columns([2, 2, 2])
                cols_header[0].markdown('**Field**')
                cols_header[1].markdown('**Current Value**')
                cols_header[2].markdown('**Proposed Value**')
                
                for field, current_val, proposed_val in changes_to_display:
                    cols_row = st.columns([2, 2, 2])
                    cols_row[0].markdown(field)
                    cols_row[1].markdown(f'`{current_val}`')
                    cols_row[2].markdown(f'<span style="color:#4CAF50; font-weight:bold;">`{proposed_val}`</span>', unsafe_allow_html=True)
                
                st.markdown("---")
            else:
                st.info("No fields were selected for update.")

            # Table 2: All other record details (not being changed)
            remaining_details_to_display = []
            all_fields = list(selected_record.index)
            change_fields = [provider_mapping[col] for col in approved_df_cols] + ["ID"]
            
            for field in all_fields:
                if field not in change_fields:
                    remaining_details_to_display.append([field, selected_record.get(field)])
            
            if not is_new_record and remaining_details_to_display:
                st.markdown("**Other profile details of the account (not changing):**")
                remaining_df = pd.DataFrame(remaining_details_to_display, columns=["Field", "Value"])
                st.dataframe(remaining_df, hide_index=True, use_container_width=True)

            if not is_new_record:
                st.markdown("---")
            # --- END MODIFIED ---
            
            # Use st.columns for horizontal buttons
            col1, col2 = st.columns([1, 1])
            confirm_btn_label = "Yes, Insert" if is_new_record else "Yes, Update"
            with col1:
                if st.button(confirm_btn_label, key="confirm_yes"):
                    approved_df_cols = st.session_state.get('approved_cols', [])
                    selected_id = st.session_state.selected_hcp_id
                                        
                    if not approved_df_cols:
                        st.info("No fields were selected. Please go back and select fields.")
                        st.session_state.show_confirm_dialog = False
                        st.rerun()
                    else:
                        spinner_text = "Inserting record in Snowflake..." if is_new_record else "Updating record in Snowflake..."
                        with st.spinner(spinner_text):
                            try:
                                db_column_map = {
                                    "Name": "LAST_NM", "Address Line1": "ADDRESS1", "Address Line2": "ADDRESS2",
                                    "City": "CITY", "State": "STATE", "ZIP": "ZIP"
                                }
                                assignments = {}
                                proposed_record = st.session_state.proposed_record
                                columns_list = []

                                for col_name in approved_df_cols:
                                    db_col_name = db_column_map.get(col_name)
                                    if db_col_name:
                                        new_value = proposed_record.get(col_name)
                                        if hasattr(new_value, 'item'): 
                                            new_value = new_value.item()
                                        assignments[db_col_name] = new_value
                                        columns_list.append(db_col_name)

                                DATABASE, SCHEMA, YOUR_TABLE_NAME = "CORTEX_ANALYST_HCK", "PUBLIC", "NPI"
                                target_table = session.table(f'"{DATABASE}"."{SCHEMA}"."{YOUR_TABLE_NAME}"')
                                
                                if is_new_record:
                                    # Get max ID and add 1 for new record
                                    # NPI table uses numeric IDs
                                    max_id_result = session.sql(f'SELECT COALESCE(MAX(ID), 0) AS MAX_ID FROM "{DATABASE}"."{SCHEMA}"."{YOUR_TABLE_NAME}"').collect()
                                    max_id = max_id_result[0].MAX_ID if max_id_result[0].MAX_ID else 0
                                    new_id = int(max_id) + 1
                                    # Add ID to columns and assignments
                                    columns_list.insert(0, "ID")
                                    assignments["ID"] = new_id
                                    
                                    # INSERT new record using SQL - ID is numeric, others are strings
                                    col_names = ", ".join(columns_list)
                                    col_values_list = []
                                    for c in columns_list:
                                        if c == "ID":
                                            col_values_list.append(str(assignments[c]))  # Numeric, no quotes
                                        elif assignments[c] is not None:
                                            col_values_list.append(f"'{str(assignments[c])}'")
                                        else:
                                            col_values_list.append("NULL")
                                    col_values = ", ".join(col_values_list)
                                    insert_sql = f'INSERT INTO "{DATABASE}"."{SCHEMA}"."{YOUR_TABLE_NAME}" ({col_names}) VALUES ({col_values})'
                                    session.sql(insert_sql).collect()
                                    cols_str = ", ".join(columns_list)
                                    custom_message = f"New record inserted successfully with ID: {new_id}. Columns: {cols_str}."
                                    st.session_state.show_popup = True
                                    st.session_state.popup_message_info = { 'type': 'insert_success', 'id': new_id, 'message': custom_message }
                                else:
                                    # UPDATE existing record
                                    update_result = target_table.update(assignments, col("ID") == selected_id)
                                    if update_result.rows_updated > 0:
                                        cols_str = ", ".join(columns_list)
                                        custom_message = f"Record for ID: {selected_id} updated successfully. Changed columns: {cols_str}."
                                        st.session_state.show_popup = True
                                        st.session_state.popup_message_info = { 'type': 'update_success', 'id': selected_id, 'message': custom_message }
                                    else:
                                        st.warning(f"Record for ID {selected_id} was not found for update.")
                                        st.session_state.show_confirm_dialog = False
                                        st.rerun()
                            except Exception as e:
                                st.error(f"An error occurred: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                                st.session_state.show_confirm_dialog = False
                                st.stop()  # Stop instead of rerun to see the error
                                
                            st.session_state.show_confirm_dialog = False
                            st.rerun()
            with col2:
                if st.button("Cancel", key="confirm_cancel"):
                    st.session_state.show_confirm_dialog = False
                    st.rerun()
        return
    
    # Check for primary update confirmation dialog
    if st.session_state.get('show_primary_confirm_dialog'):
        # Check if this is a new HCP from bypass flow
        is_new_hcp = st.session_state.selected_hcp_id == 'empty_record'
        
        with dialog_placeholder.container():
            # Different warning text based on whether this is a new HCP
            if is_new_hcp:
                st.warning(
                    "This is a new provider record. Setting primary affiliation will:\n"
                    "1. Create a new affiliation record in HCP_HCO_AFFILIATION table\n"
                    "2. Set this as the primary affiliation in the NPI table",
                    icon="⚠️"
                )
            else:
                st.warning("Are you sure you want to change the primary affiliation? This will update the main record.", icon="⚠️")
            
            # --- MODIFIED: Display primary affiliation change in a vertical table ---
            current_primary_id = selected_record.get("PRIMARY_AFFL_HCO_ACCOUNT_ID")
            current_primary_name_query = session.sql(f"SELECT HCO_NAME FROM HCP_HCO_AFFILIATION WHERE HCO_ID = '{current_primary_id}'").collect() if current_primary_id else None
            current_primary_name = current_primary_name_query[0].HCO_NAME if current_primary_name_query and not current_primary_name_query[0].HCO_NAME is None else "N/A"
            
            new_primary_id = st.session_state.primary_hco_id
            # Get HCO data from session state (stored when "Set as Primary" was clicked)
            new_hco_data = st.session_state.get('primary_hco_data', {})
            new_primary_name = new_hco_data.get('HCO NAME', new_hco_data.get('HCO_Name', 'N/A'))
            
            # Check if this is AI-generated (will need a new ID)
            is_ai_generated_hco = (str(new_primary_id).startswith('ai_generated_') or 
                                   new_hco_data.get('SOURCE') == 'Generated by AI' or
                                   new_hco_data.get('SOURCE', '').lower() == 'generated by ai')
            
            # For AI-generated, show that a new ID will be created
            if is_ai_generated_hco:
                proposed_hco_display = f"NEW ID (will be generated) - {new_primary_name}"
            else:
                proposed_hco_display = f"ID: {new_primary_id} ({new_primary_name})"

            primary_change_df = pd.DataFrame({
                "Field": ["ID", "Name", "Current Primary HCO", "Proposed Primary HCO"],
                "Value": [
                    selected_record.get('ID'),
                    selected_record.get('NAME'),
                    f"ID: {current_primary_id} ({current_primary_name})" if current_primary_id else "None",
                    proposed_hco_display
                ]
            })
            
            if is_new_hcp or is_ai_generated_hco:
                # Add row indicating new affiliation will be created
                new_row = pd.DataFrame({"Field": ["New Affiliation Record"], "Value": ["Will be created in HCP_HCO_AFFILIATION table with auto-generated ID"]})
                primary_change_df = pd.concat([primary_change_df, new_row], ignore_index=True)
            
            st.dataframe(primary_change_df.set_index('Field'), use_container_width=True)
            # --- END MODIFIED ---

            col1, col2 = st.columns([1, 1])
            
            # Different button text based on action
            # Check if AI-generated: either ID starts with 'ai_generated_' OR SOURCE is 'Generated by AI'
            is_ai_generated = (str(new_primary_id).startswith('ai_generated_') or 
                              new_hco_data.get('SOURCE') == 'Generated by AI' or
                              new_hco_data.get('SOURCE', '').lower() == 'generated by ai')
            if is_new_hcp or is_ai_generated:
                confirm_btn_text = "Yes, Create Affiliation & Set Primary"
            else:
                confirm_btn_text = "Yes, Set Primary"
            
            with col1:
                if st.button(confirm_btn_text, key="confirm_primary_yes"):
                    new_primary_id = st.session_state.primary_hco_id
                    selected_id = st.session_state.selected_hcp_id
                    
                    # Get HCP NPI from proposed record or session state
                    hcp_npi = st.session_state.get('proposed_record', {}).get('NPI', '')
                    if not hcp_npi:
                        hcp_npi = selected_record.get('NPI', '')
                    
                    # Check if this is an AI-generated HCO (needs to be inserted first)
                    # AI-generated if: ID starts with 'ai_generated_' OR SOURCE is 'Generated by AI'
                    is_ai_generated = (str(new_primary_id).startswith('ai_generated_') or 
                                      new_hco_data.get('SOURCE') == 'Generated by AI' or
                                      new_hco_data.get('SOURCE', '').lower() == 'generated by ai')
                    
                    spinner_text = "Creating affiliation and setting primary..." if (is_new_hcp or is_ai_generated) else "Updating primary affiliation in Snowflake..."
                    with st.spinner(spinner_text):
                        try:
                            success = True
                            final_hco_id = new_primary_id  # Will be updated if AI-generated
                            
                            # For AI-generated HCO (new or existing HCP), first insert the affiliation record
                            if is_ai_generated and new_hco_data:
                                # Insert and get the new HCO_ID
                                result = insert_affiliation_record(session, selected_id, hcp_npi, new_hco_data, generate_new_id=True)
                                if result is not None:
                                    final_hco_id = result  # Use the newly generated ID
                                    st.toast(f"✅ Affiliation record created with HCO ID: {final_hco_id}", icon="✅")
                                else:
                                    success = False
                                    st.error("Failed to create affiliation record.")
                            elif is_new_hcp and new_hco_data:
                                # For new HCP with existing HCO (not AI-generated)
                                if not check_affiliation_exists(session, hcp_npi, new_primary_id):
                                    success = insert_affiliation_record(session, selected_id, hcp_npi, new_hco_data)
                                    if success:
                                        st.toast("✅ Affiliation record created successfully!", icon="✅")
                                else:
                                    st.info("Affiliation record already exists.")
                            
                            # Now update the primary affiliation in NPI table with the final HCO ID
                            if success:
                                npi_table = session.table("NPI")
                                update_assignments = {"PRIMARY_AFFL_HCO_ACCOUNT_ID": final_hco_id}
                                update_result = npi_table.update(update_assignments, col("ID") == selected_id)
                                if update_result.rows_updated > 0:
                                    st.session_state.show_popup = True
                                    st.session_state.popup_message_info = { 'type': 'primary_success', 'hco_id': final_hco_id }
                                    st.session_state.show_primary_confirm_dialog = False
                                    st.session_state.primary_hco_data = None  # Clear stored HCO data
                                    st.rerun()
                                else:
                                    st.warning("Could not find the main HCP record to update. Please ensure the record was inserted first.")
                                    st.session_state.show_primary_confirm_dialog = False
                                    st.rerun()
                        except Exception as e:
                            st.error(f"An error occurred during the update: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                            st.session_state.show_primary_confirm_dialog = False
            with col2:
                if st.button("Cancel", key="confirm_primary_cancel"):
                    st.session_state.show_primary_confirm_dialog = False
                    st.session_state.primary_hco_data = None  # Clear stored HCO data
                    st.rerun()
        return
#end of Placeholder for a potential dialog to display over the main content

    with st.spinner("🚀 Contacting AI Assistant for Data Enrichment..."):
        # Get user's search query for web search flow (when no record exists in DB)
        user_search_query = st.session_state.get('web_search_query', None)
        api_response = get_enriched_data_from_llm(session, selected_hcp_df, search_query=user_search_query)
        
        # Safely extract data with fallbacks
        if not isinstance(api_response, dict):
            st.error("API response is not a dictionary. Cannot proceed.")
            st.stop()
        
        # Try different key names for HCP data
        hcp_data = api_response.get('hcp_data', api_response.get('hco_data', []))
        proposed_hcp_data_df = pd.DataFrame(hcp_data)
        
        # Try different key names for affiliation data
        affiliation_data = api_response.get('hcp_affiliation_data', api_response.get('hco_affiliation_data', []))
        # Handle case where arrays have different lengths (e.g., NPI has 1 item, others have 2)
        if isinstance(affiliation_data, dict):
            # Find the max length among all arrays
            max_len = max((len(v) if isinstance(v, list) else 1) for v in affiliation_data.values()) if affiliation_data else 0
            # Pad shorter arrays with their first value or None
            for key, val in affiliation_data.items():
                if isinstance(val, list) and len(val) < max_len:
                    # Pad with the first value if available, otherwise None
                    pad_value = val[0] if val else None
                    affiliation_data[key] = val + [pad_value] * (max_len - len(val))
        proposed_hcp_affiliation_data_df = pd.DataFrame(affiliation_data) if affiliation_data else pd.DataFrame()

    try:
        if current_df.empty or proposed_hcp_data_df.empty:
            st.warning("Could not generate a comparison report.")
            st.stop()
    except AttributeError:
        st.error("One of the dataframes is invalid. Please check the data source.")
        st.stop()

    # Handle ID - may be 'N/A' for empty record flow
    raw_id = current_df['ID'].iloc[0]
    try:
        selected_id = int(raw_id) if raw_id != 'N/A' and pd.notna(raw_id) else 'N/A'
    except (ValueError, TypeError):
        selected_id = 'N/A'
    current_record = current_df.iloc[0]
    proposed_hcp_data_record = proposed_hcp_data_df.iloc[0]


#session state for proposed record

    st.session_state.proposed_hcp_data_record = proposed_hcp_data_record
    
    if "demographic_expander_state" not in st.session_state:
        st.session_state.demographic_expander_state = False
    
    provider_mapping = { "Name": "Name", "Address Line 1": "Address Line1", "Address Line 2": "Address Line2", "City": "City", "State": "State", "ZIP Code": "ZIP" }
    
    for field_label, col_name in provider_mapping.items():
        if st.session_state.get(f"approve_{selected_id}_{col_name}", False):
            st.session_state.demographic_expander_state = True
            break
#end of session state for proposed record

    #st.header(f"Comparing for ID: {selected_id} | {current_record.get('Name', '')} | NPI: {current_record.get('NPI', 'N/A')}")
    st.markdown(
        f"<h5>Comparing for ID: {selected_id} | {current_record.get('Name', '')} | NPI: {current_record.get('NPI', 'N/A')}</h5>", 
        unsafe_allow_html=True
    )

    
#provider_info_change
    provider_info_title = f"Demographic information of : {current_record.get('Name', 'N/A')} (NPI: {current_record.get('NPI', 'N/A')})"
    
    with st.expander(provider_info_title, expanded=st.session_state.demographic_expander_state): 
        
        header_cols = st.columns([2, 2, 2, 1.5, 2.5, 1])
        headers = ["Field", "Current", "Proposed", "Confidence", "Sources", "Approve"]
        for column_obj, header_name in zip(header_cols, headers):
            column_obj.markdown(f'<div class="report-header">{header_name}</div>', unsafe_allow_html=True)
        
        provider_mapping = { "Name": "Name", "Address Line 1": "Address Line1", "Address Line 2": "Address Line2", "City": "City", "State": "State", "ZIP Code": "ZIP" }

        for field_label, col_name in provider_mapping.items():
            current_val = current_record.get(col_name, "")
            proposed_val = proposed_hcp_data_record.get(col_name, "")
            score = proposed_hcp_data_record.get(f"{col_name}_Score", 0)
            
            source_data = proposed_hcp_data_record.get(f"{col_name}_Source")
            source_display = "N/A"
            if isinstance(source_data, list) and source_data:
                generic_urls = []
                for url in source_data:
                    try:
                        parsed_url = urlparse(str(url))
                        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                        if base_url not in generic_urls:
                            generic_urls.append(base_url)
                    except:
                        if str(url) not in generic_urls:
                            generic_urls.append(str(url))
                source_display = ", ".join(generic_urls)
            elif source_data and not isinstance(source_data, list):
                try:
                    parsed_url = urlparse(str(source_data))
                    source_display = f"{parsed_url.scheme}://{parsed_url.netloc}"
                except: source_display = str(source_data)
                
            score_percent = score
            
            
            row_cols = st.columns([2, 2, 2, 1.5, 2.5, 1])
            row_cols[0].markdown(f'<div class="cell-content" style="font-weight: bold;">{field_label}</div>', unsafe_allow_html=True)
            row_cols[1].markdown(f'<div class="cell-content">{current_val}</div>', unsafe_allow_html=True)

            if st.session_state.get(f"approve_{selected_id}_{col_name}", False):
                row_cols[2].markdown(f'<div class="cell-content report-proposed-column" style="font-weight: bold; color: #4CAF50;">{proposed_val}</div>', unsafe_allow_html=True)
            else:
                row_cols[2].markdown(f'<div class="cell-content report-proposed-column">{proposed_val}</div>', unsafe_allow_html=True)
            
            row_cols[3].markdown(f'<div class="cell-content">{score_percent}</div>', unsafe_allow_html=True)
            row_cols[4].markdown(f'<div class="cell-content">{source_display.replace(", ", "<br>")}</div>', unsafe_allow_html=True)
            with row_cols[5]:
                st.markdown('<div class="cell-content checkbox-container">', unsafe_allow_html=True)
                st.checkbox("", key=f"approve_{selected_id}_{col_name}", label_visibility="collapsed")
                st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        _, btn_col = st.columns([5, 1])
        is_new_record = selected_id == 'N/A' or st.session_state.selected_hcp_id == 'empty_record'
        btn_label = "Insert Record 💾" if is_new_record else "Update Record 💾"
        with btn_col:
            if st.button(btn_label, type="primary", key=f"update_btn_{selected_id}"):
                approved_df_cols = []
                for field_label, col_name in provider_mapping.items():
                    checkbox_key = f"approve_{selected_id}_{col_name}"
                    if st.session_state.get(checkbox_key, False):
                        approved_df_cols.append(col_name)

                if approved_df_cols:
                    st.session_state.show_confirm_dialog = True
                    st.session_state.approved_cols = approved_df_cols
                    st.session_state.proposed_record = proposed_hcp_data_record.to_dict() if hasattr(proposed_hcp_data_record, 'to_dict') else proposed_hcp_data_record
                    st.rerun()
                else:
                    st.info(f"No fields were selected for ID {selected_id}.")

    st.markdown("<hr style='margin-top: 0; margin-bottom: 0; border-top: 1px solid #ccc;'>", unsafe_allow_html=True)
    
    hco_affiliation_title = f"HCO Affiliation information of : {current_record.get('Name', 'N/A')} (NPI: {current_record.get('NPI', 'N/A')})"
    
    with st.expander(hco_affiliation_title, expanded=False):
        
        # Updated headers to include Priority and Reason columns
        hco_headers = ["Status", "SOURCE", "HCO ID", "HCO NAME", "HCO ADDRESS", "HCO CITY", "HCO STATE", "HCO ZIP", "Priority", "Reason"]
        header_cols = st.columns([1.5, 1.5, 1.2, 2.5, 2, 1.2, 1.2, 1.2, 0.8, 1])
        for col_obj, header_name in zip(header_cols, hco_headers):
            col_obj.markdown(f"**{header_name}**")
        
        primary_id_val = selected_record.get("PRIMARY_AFFL_HCO_ACCOUNT_ID")
        true_primary_hco_id = int(primary_id_val) if pd.notna(primary_id_val) else None
        
        hcp_npi = current_record.get("NPI")
        db_affiliations_df = pd.DataFrame()
        if hcp_npi:
            query = f"SELECT * FROM HCP_HCO_AFFILIATION WHERE HCP_NPI = '{hcp_npi}'"
            db_affiliations_df = session.sql(query).to_pandas()

        # Build ai_found_hcos from proposed_hcp_affiliation_data_df
        ai_found_hcos = []
        # Get HCP name to filter out affiliations that match the HCP's own name
        hcp_name = current_record.get('Name', current_record.get('NAME', '')).upper().strip()
        hcp_name_parts = [p.strip() for p in hcp_name.replace(',', ' ').replace('.', ' ').split() if len(p.strip()) > 2]
        
        if not proposed_hcp_affiliation_data_df.empty:
            for index, row in proposed_hcp_affiliation_data_df.iterrows():
                hco_name = row.get('HCO_Name')
                if pd.notna(hco_name) and str(hco_name).strip() != "":
                    hco_name_upper = str(hco_name).upper().strip()
                    
                    # Skip if HCO name contains the HCP's name (likely the HCP's own practice)
                    is_hcp_own_practice = False
                    if hcp_name_parts:
                        matching_parts = sum(1 for part in hcp_name_parts if part in hco_name_upper)
                        if matching_parts >= len(hcp_name_parts) / 2:
                            is_hcp_own_practice = True
                    
                    if not is_hcp_own_practice:
                        ai_found_hcos.append({
                            "HCO ID": row.get('HCO_ID'),
                            "HCO NAME": hco_name, "HCO NPI": row.get('NPI'),
                            "HCO ADDRESS": row.get('HCO_Address1', ''),
                            "HCO CITY": row.get('HCO_City', ''), "HCO STATE": row.get('HCO_State', ''), "HCO ZIP": row.get('HCO_ZIP', ''),
                        })

        all_affiliations = {}
        if not db_affiliations_df.empty:
            for index, row in db_affiliations_df.iterrows():
                hco_id = row.get('HCO_ID')
                if not hco_id: continue
                all_affiliations[hco_id] = {
                    "SOURCE": "HCOS data", "HCP_NPI": row.get('HCP_NPI'),
                    "HCO ID": hco_id, "HCO NAME": row.get('HCO_NAME'),
                    "HCO ADDRESS": f"{row.get('HCO_ADDRESS1', '')}, {row.get('HCO_ADDRESS2', '')}".strip(", "),
                    "HCO CITY": row.get('HCO_CITY'), "HCO STATE": row.get('HCO_STATE'), "HCO ZIP": row.get('HCO_ZIP'),
                }

        for idx, hco in enumerate(ai_found_hcos):
            hco_id = hco.get('HCO ID')
            hco["SOURCE"] = "Generated by AI"
            hco["HCP_NPI"] = None
            # Use a unique key for AI-generated entries (use index if no HCO ID)
            key = hco_id if hco_id and hco_id not in all_affiliations else f"ai_generated_{idx}"
            all_affiliations[key] = hco
        
        if not all_affiliations:
            st.info("No HCO affiliations were found.")
        else:
            # Create a cache key based on selected HCP ID and affiliation keys
            selected_hcp_id_for_cache = current_data_dict.get('ID', '')
            affiliation_keys = sorted([str(k) for k in all_affiliations.keys()])
            cache_key = f"{selected_hcp_id_for_cache}_{'_'.join(affiliation_keys[:5])}"
            
            # Initialize priority cache in session state if not exists
            if 'priority_rankings_cache' not in st.session_state:
                st.session_state.priority_rankings_cache = {}
            
            # Check if we already have cached priority rankings for this HCP
            priority_rankings = st.session_state.priority_rankings_cache.get(cache_key, {})
            
            # Show button to analyze priorities if not already analyzed
            if not priority_rankings:
                st.markdown("---")
                col1, col2, col3 = st.columns([2, 2, 2])
                with col2:
                    if st.button("🎯 Analyze Priority Order with AI", key=f"analyze_priorities_{selected_hcp_id_for_cache}", use_container_width=True):
                        st.session_state[f'analyze_priorities_clicked_{cache_key}'] = True
                        st.rerun()
                
                # Check if button was clicked to trigger analysis
                if st.session_state.get(f'analyze_priorities_clicked_{cache_key}', False):
                    affiliations_list = list(all_affiliations.items())
                    
                    # Show prominent status message
                    status_placeholder = st.empty()
                    status_placeholder.info("🤖 **AI Analysis in Progress**\n\nSending affiliation data to LLM to determine priority order and reasoning...")
                    
                    with st.spinner("⏳ Fetching priority rankings and reasons from AI..."):
                        priority_rankings = get_affiliation_priorities_from_llm(
                            session, 
                            current_data_dict, 
                            affiliations_list
                        )
                    
                    # Clear the status message after completion
                    status_placeholder.empty()
                    st.toast("✅ AI analysis complete! Affiliations ranked by priority.", icon="🎯")
                    
                    # Cache the results and clear the click state
                    st.session_state.priority_rankings_cache[cache_key] = priority_rankings
                    st.session_state[f'analyze_priorities_clicked_{cache_key}'] = False
                    st.rerun()
            
            # Store priority rankings in session state for popup access
            st.session_state.priority_reasons = priority_rankings
            
            # Sort affiliations by LLM priority if available, otherwise keep original order
            def get_priority_for_sort(item):
                if not priority_rankings:
                    return 0  # Keep original order if no rankings
                try:
                    return int(priority_rankings.get(str(item[0]), {}).get("priority", 999))
                except (ValueError, TypeError):
                    return 999
            
            sorted_affiliations = sorted(all_affiliations.items(), key=get_priority_for_sort)
            
            for hco_id, hco_data in sorted_affiliations:
                # Match header columns: 10 columns total
                row_cols = st.columns([1.5, 1.5, 1.2, 2.5, 2, 1.2, 1.2, 1.2, 0.8, 1])
                
                is_primary = False
                try:
                    is_primary = hco_id != "N/A" and true_primary_hco_id is not None and int(hco_id) == true_primary_hco_id
                except (ValueError, TypeError):
                    pass
                
                # Check if this is a new HCP from bypass flow
                is_new_hcp = st.session_state.selected_hcp_id == 'empty_record'
                
                with row_cols[0]:
                    if is_primary:
                        st.markdown("✅ **Primary**")
                    else:
                        # Different button text for new HCP
                        btn_text = "Add & Set Primary" if is_new_hcp else "Set as Primary"
                        if st.button(btn_text, key=f"set_primary_{hco_id}"):
                            st.session_state.show_primary_confirm_dialog = True
                            st.session_state.primary_hco_id = hco_id
                            # Store the full HCO data for insertion (needed for bypass flow)
                            st.session_state.primary_hco_data = hco_data
                            st.rerun()
                
                source = hco_data.get("SOURCE", "")
                is_ai_source = (source == "Generated by AI")
                row_cols[1].write(source)
                
                if is_ai_source:
                    row_cols[2].write("")
                else:
                    row_cols[2].write(str(hco_data.get("HCO ID", "")))
        
                row_cols[3].write(hco_data.get("HCO NAME", ""))
                row_cols[4].write(hco_data.get("HCO ADDRESS", ""))
                row_cols[5].write(hco_data.get("HCO CITY", ""))
                row_cols[6].write(hco_data.get("HCO STATE", ""))
                row_cols[7].write(hco_data.get("HCO ZIP", ""))
                
                # Priority column - show "-" if not analyzed yet
                priority_info = priority_rankings.get(str(hco_id), {"priority": "-", "reason": "N/A"})
                row_cols[8].write(str(priority_info.get("priority", "-")))
                
                # Reason button column - only show if priorities have been analyzed
                with row_cols[9]:
                    if priority_rankings:
                        reason_key = f"reason_{hco_id}"
                        if st.button("ℹ️", key=reason_key, help="Click to see why this priority was assigned"):
                            st.session_state.show_reason_popup = True
                            st.session_state.reason_popup_data = {
                                "hco_name": hco_data.get("HCO NAME", "Unknown"),
                                "priority": priority_info.get("priority", "-"),
                                "reason": priority_info.get("reason", "No reason available")
                            }
                            st.rerun()
                    else:
                        st.write("-")

#----------end of provider_info_change



# --- MAIN DATA STEWARD APP PAGE FUNCTION ---
def render_main_page(session):
    st.header("🤖 Data Stewardship Assistant")
    st.html("<style> .main {overflow: hidden}</style>")
    with st.sidebar:
        st.subheader("About this App")
        st.info("This app is the single source of truth for Data Stewards, leveraging Snowflake Cortex AI to seamlessly consolidate the latest demographic and affiliation updates from multiple web sources and the Enterprise Data Warehouse, ensuring master data accuracy.")
     # --- Custom CSS for the detail section ---
    st.markdown("""
        <style>
            .detail-key { font-weight: bold; color: #4F8BE7; margin-top: 0.5rem;}
            .detail-value { padding-bottom: 0.5rem; }
        </style>
    """, unsafe_allow_html=True)

    DATABASE = "CORTEX_ANALYST_HCK"
    SCHEMA = "PUBLIC"
    STAGE = "HACKATHON"
    FILE = "HCK_MODEL.yaml"

    def send_message(prompt: str) -> dict:
        """
        Sends a message to Snowflake Cortex Analyst API using REST.
        For Streamlit Community Cloud, this uses the requests library with JWT auth.
        """
        # Build the API URL using account from secrets
        account = st.secrets["snowflake"]["account"]
        # Handle account identifier format (remove region suffix if present for URL)
        account_url = account.replace("_", "-").replace(".", "-")
        api_url = f"https://{account_url}.snowflakecomputing.com/api/v2/cortex/analyst/message"
        
        # Get a fresh token from the session
        token = session.connection.rest.token
        
        headers = {
            "Authorization": f"Snowflake Token=\"{token}\"",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        request_body = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "semantic_model_file": f"@{DATABASE}.{SCHEMA}.{STAGE}/{FILE}",
        }
        
        resp = requests.post(api_url, headers=headers, json=request_body, timeout=30)
        
        if resp.status_code < 400:
            return resp.json()
        else:
            raise Exception(f"Failed request with status {resp.status_code}: {resp.text}")

    def process_message(prompt: str) -> None:
        st.session_state.messages.clear()
        st.session_state.results_df = None
        st.session_state.selected_hcp_id = None
        st.session_state.messages.append(
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        )
        with st.spinner("Generating response..."):
            try:
                response = send_message(prompt=prompt)
                question_item = {"type": "text", "text": prompt.strip()}
                response["message"]["content"].insert(0, question_item)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response["message"]["content"]}
                )
            except Exception as e:
                st.error(f"error occured: {str(e)}")

    def display_interpretation(content: list):
        if not content:
            return
        user_query = content[0].get("text")
        st.markdown(f'You searched : "{user_query}"')

        if len(content) > 1 and content[1].get("type") == "text":
            interpretation_full_text = content[1].get("text")
            prefix_to_remove = "This is our interpretation of your question:"
            interpretation_clean = interpretation_full_text.replace(
                prefix_to_remove, ""
            ).strip()
            st.markdown(
                f'This is our interpretation of your question : "{interpretation_clean}"'
            )

    def display_results_table(content: list):
        sql_item_found = False
        for item in content:
            if item["type"] == "sql":
                sql_item_found = True
                with st.spinner("Running SQL..."):
                    df = session.sql(item["statement"]).to_pandas()
                    if not df.empty:
                        st.session_state.results_df = df
                        st.write("Please select a record from the table to proceed:")
                        # Define Column Sizes & Heading Names
                        cols = st.columns((0.8, 0.8, 1.2, 1, 2, 1, 0.5))
                        headers = ["Select", "ID", "Name", "NPI", "Address", "City", "State"]

                        for col_header, header_name in zip(cols, headers):
                            col_header.markdown(f"**{header_name}**")

                        for index, row in df.iterrows():
                            row_id = row.get("ID")
                            if row_id is None:
                                continue
                            is_selected = row_id == st.session_state.get("selected_hcp_id")
                            row_cols = st.columns((0.8, 0.8, 1.2, 1, 2, 1, 0.5))

                            if is_selected:
                                row_cols[0].write("🔘")
                            else:
                                if row_cols[0].button("", key=f"select_{row_id}"):
                                    st.session_state.selected_hcp_id = row_id
                                    st.rerun()

                            row_cols[1].write(row_id)
                            row_cols[2].write(row.get("NAME", ""))
                            row_cols[3].write(row.get("NPI", "N/A"))
                            row_cols[4].write(row.get("ADDRESS1", "N/A"))
                            row_cols[5].write(row.get("CITY", "N/A"))
                            row_cols[6].write(row.get("STATE", "N/A"))
                    else:
                        st.info("We couldn't find any records matching your search.", icon="ℹ️")
                        st.markdown("")
                        if st.button("🔍 Still want to proceed with Web Search?", type="primary"):
                            # Create a default empty record for enrichment
                            st.session_state.empty_record_for_enrichment = {
                                'ID': 'N/A',
                                'NAME': '',
                                'NPI': '',
                                'ADDRESS1': '',
                                'ADDRESS2': '',
                                'CITY': '',
                                'STATE': '',
                                'ZIP': '',
                                'COUNTRY': '',
                                'PRIMARY_AFFL_HCO_ACCOUNT_ID': None
                            }
                            # Store the search query for web search context
                            st.session_state.web_search_query = st.session_state.get('last_prompt', '')
                            st.session_state.selected_hcp_id = 'empty_record'
                            st.session_state.current_view = "enrichment_page"
                            st.rerun()
        if not sql_item_found:
            st.info("The assistant did not return a SQL query for this prompt. It may be a greeting or a clarifying question.")
            st.markdown("")
            if st.button("🔍 Still want to proceed with Web Search?", key="web_search_no_sql", type="primary"):
                # Create a default empty record for enrichment
                st.session_state.empty_record_for_enrichment = {
                    'ID': 'N/A',
                    'NAME': '',
                    'NPI': '',
                    'ADDRESS1': '',
                    'ADDRESS2': '',
                    'CITY': '',
                    'STATE': '',
                    'ZIP': '',
                    'COUNTRY': '',
                    'PRIMARY_AFFL_HCO_ACCOUNT_ID': None
                }
                # Store the search query for web search context
                st.session_state.web_search_query = st.session_state.get('last_prompt', '')
                st.session_state.selected_hcp_id = 'empty_record'
                st.session_state.current_view = "enrichment_page"
                st.rerun()

    # --- MAIN INPUT LOGIC ---
    freeze_container = st.container(border=True)
    with freeze_container:
        user_input_text = st.chat_input("Search for an Account")
        current_prompt = user_input_text

        if current_prompt and current_prompt != st.session_state.get("last_prompt"):
            process_message(prompt=current_prompt)
            st.session_state.last_prompt = current_prompt

   # --- DISPLAY LOGIC (Vertical Flow) ---
    if st.session_state.messages:
        assistant_messages = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
        if assistant_messages:
            st.markdown("---")
            display_interpretation(content=assistant_messages[-1]["content"])

            # 1. Search Results Table (Full Width)
            response_container = st.container(border=True)
            with response_container:
                st.subheader("Search Results")
                display_results_table(content=assistant_messages[-1]["content"])

            # Helper to safely get and format value
            def get_safe_value(record, key):
                value = record.get(key)
                return str(value) if pd.notna(value) and value is not None else 'N/A'
            

 # 2. Selected Record Details (Only appears when selected_hcp_id is set)
            if st.session_state.get("selected_hcp_id") and st.session_state.get("results_df") is not None:
                
                selected_record_df = st.session_state.results_df[
                    st.session_state.results_df["ID"] == st.session_state.selected_hcp_id
                ]
                
                if not selected_record_df.empty:
                    
                    selected_record = selected_record_df.iloc[0]
                    
                    # --- Start Two-Column Layout for Details Sections (Side-by-Side Below Search) ---
                    details_col_left, details_col_right = st.columns(2)
                    
                    # --- Left Detail Column: Current Demographic Details ---
                    with details_col_left:
                        st.subheader("Current Demographic Details")
                        
                        # This border container holds the custom 2x2 layout
                        with st.container(border=True):
                            
                            # ID and Name in the required structure (single line)
                            hcp_id = get_safe_value(selected_record, 'ID')
                            hcp_name = get_safe_value(selected_record, 'NAME')
                            st.markdown(f'**ID:** {hcp_id} - {hcp_name}', unsafe_allow_html=True)
                            #st.markdown("---")
                            st.markdown("<hr style='margin-top: 0; margin-bottom: 0; border-top: 1px solid #ccc;'>", unsafe_allow_html=True)

                            # Define the fields for the new two-column layout
                            identity_fields = [
                                ("Prefix", "PREFIX"),
                                ("First Name", "FIRST_NM"),
                                ("Middle Name", "MIDDLE_NM"),
                                ("Last Name", "LAST_NM"),
                                ("Suffix", "SUFFIX"),
                                ("Degree", "DEGREE"),
                            ]
                            address_fields = [
                                ("Address Line 1", "ADDRESS1"),
                                ("Address Line 2", "ADDRESS2"),
                                ("City", "CITY"),
                                ("State", "STATE"),
                                ("ZIP", "ZIP"),
                                ("Country", "COUNTRY")
                            ]

                            # Create two internal columns for the key-value pairs
                            col_identity, col_address = st.columns(2)

                            # Render Identity Fields (Left Column)
                            for label, key in identity_fields:
                                value = get_safe_value(selected_record, key)
                                col_identity.markdown(
                                    f'<div class="detail-key">{label}:</div>'
                                    f'<div class="detail-value">{value}</div>',
                                    unsafe_allow_html=True
                                )

                            # Render Address Fields (Right Column)
                            for label, key in address_fields:
                                value = get_safe_value(selected_record, key)
                                col_address.markdown(
                                    f'<div class="detail-key">{label}:</div>'
                                    f'<div class="detail-value">{value}</div>',
                                    unsafe_allow_html=True
                                )

                    # --- Right Detail Column: Primary HCO Affiliation Details ---
                    with details_col_right:
                        st.subheader("Primary HCO Affiliation Details")
                        with st.container(border=True):
                            hco_col1, hco_col2 = st.columns(2)
                            
                            hco_id_val = "N/A"
                            hco_name_val = "N/A"
                            hco_npi_val = "N/A"
                            
                            # First try PRIMARY_AFFL_HCO_ACCOUNT_ID from NPI table
                            primary_hco_id = selected_record.get("PRIMARY_AFFL_HCO_ACCOUNT_ID")
                            
                            # Get HCP identifiers for affiliation lookup
                            hcp_id = selected_record.get("ID")
                            hcp_npi = selected_record.get("NPI")
                            
                            try:
                                if pd.notna(primary_hco_id) and primary_hco_id is not None:
                                    # Use the primary HCO ID to get details
                                    hco_id_val = str(primary_hco_id)
                                    hco_query = session.sql(f"SELECT NAME, NPI FROM HCO WHERE ID = '{primary_hco_id}'").collect()
                                    if hco_query:
                                        hco_name_val = hco_query[0].NAME if hco_query[0].NAME else "N/A"
                                        hco_npi_val = str(hco_query[0].NPI) if hco_query[0].NPI else "N/A"
                                else:
                                    # Fetch from HCP_HCO_AFFILIATION table using HCP_NPI
                                    aff_query = session.sql(f"SELECT HCO_ID, HCO_NAME FROM HCP_HCO_AFFILIATION WHERE HCP_NPI = '{hcp_npi}' LIMIT 1").collect()
                                    if aff_query:
                                        hco_id_val = str(aff_query[0].HCO_ID) if aff_query[0].HCO_ID else "N/A"
                                        hco_name_val = aff_query[0].HCO_NAME if aff_query[0].HCO_NAME else "N/A"
                            except:
                                pass
                            
                            hco_col1.markdown(f'<div class="detail-key">HCO ID:</div><div class="detail-value">{hco_id_val}</div>', unsafe_allow_html=True)
                            hco_col2.markdown(f'<div class="detail-key">HCO NPI:</div><div class="detail-value">{hco_npi_val}</div>', unsafe_allow_html=True)
                            hco_col1.markdown(f'<div class="detail-key">HCO Name:</div><div class="detail-value">{hco_name_val}</div>', unsafe_allow_html=True)
                            

                    # --- End Two-Column Layout for Details Sections ---
                    st.divider()
                    
                    # --- Enrich Button (Full Width, below the detail columns) ---
                    # -----------------------------------------------------------
                    # MODIFIED SECTION STARTS HERE
                    # -----------------------------------------------------------
                    # Create two columns, with a small width for the button
                    button_col, _ = st.columns([0.2, 0.8])
                    
                    with button_col:
                        if st.button("Enrich with AI Assistant 🚀", type="primary"):
                            st.session_state.current_view = "enrichment_page"
                            st.rerun()
                    # -----------------------------------------------------------
                    # MODIFIED SECTION ENDS HERE
                    # -----------------------------------------------------------
                    
                else:
                    st.info("No data found for selected record ID.")
    

# --- PAGE ROUTER ---
# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "selected_hcp_id" not in st.session_state:
    st.session_state.selected_hcp_id = None
if "current_view" not in st.session_state:
    st.session_state.current_view = "main"
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = None
if "show_popup" not in st.session_state:
    st.session_state.show_popup = False
if "popup_message_info" not in st.session_state:
    st.session_state.popup_message_info = None
if "show_confirm_dialog" not in st.session_state:
    st.session_state.show_confirm_dialog = False
if "show_primary_confirm_dialog" not in st.session_state:
    st.session_state.show_primary_confirm_dialog = False
if "primary_hco_data" not in st.session_state:
    st.session_state.primary_hco_data = None

session = get_snowflake_session()
os.environ["PERPLEXITY_API_KEY"] = st.secrets["perplexity"]["api_key"]

client = Perplexity()

# provider_mapping = { "Name": "Name", "Address Line 1": "Address Line1", "Address Line 2": "Address Line2", "City": "City", "State": "State", "ZIP Code": "ZIP" }

class HCPData(BaseModel):
    Name: list[str]
    address_line_1: List[str] = Field(..., alias="Address Line1")
    address_line_2: List[str] = Field(..., alias="Address Line2")
    ZIP: list[str]
    City: list[str]
    State: list[str]

class HCPAffiliationData(BaseModel):
    NPI: list[str]
    HCO_ID: list[str]
    HCO_Name: list[str]
    HCO_Address1: list[str]
    HCO_City: list[str]
    HCO_State: list[str]
    HCO_ZIP: list[str]

class SearchResponse(BaseModel):
    hcp_data: HCPData
    hcp_affiliation_data: HCPAffiliationData
    

def get_consolidated_data_for_hcp(hcp_data, model_name="sonar", use_pro_search=False, search_query=None):
    # Extract key info for better search - handle both dict and pandas Series
    if hasattr(hcp_data, 'to_dict'):
        hcp_data = hcp_data.to_dict()
    
    # Helper to get value with fallback
    def get_hcp_val(key):
        if isinstance(hcp_data, dict):
            val = hcp_data.get(key, '')
            return val
        return str(hcp_data)
    
    # Use search_query as the name if NAME field is empty (for Web Search flow)
    hcp_name = get_hcp_val('NAME') or search_query or ''
    hcp_npi = get_hcp_val('NPI')
    hcp_address1 = get_hcp_val('ADDRESS1')
    hcp_city = get_hcp_val('CITY')
    hcp_state = get_hcp_val('STATE')
    hcp_zip = get_hcp_val('ZIP')
    
    user_query = f"""
    You are a healthcare data research specialist. Search the web thoroughly for information about this US healthcare provider:
    
    **Provider to Research:**
    - Name: {hcp_name}
    - NPI: {hcp_npi}
    - Address Line 1: {hcp_address1}
    - City: {hcp_city}
    - State: {hcp_state}
    - ZIP: {hcp_zip}

    **IMPORTANT INSTRUCTIONS:**
    1. You MUST search the web and find COMPLETE information for ALL fields requested below
    2. Do NOT return "N/A" for address fields if the provider exists - search harder to find the actual address
    3. For affiliated organizations, search their official website, NPI Registry, or healthcare directories to find their complete address
    4. If you find an affiliated HCO name, you MUST also find and return their complete address

    **Part 1 - Provider Demographics (verify/update from web sources):**
    Search for the current, verified information about "{hcp_name}":
    - Name: Full name of the doctor/provider
    - Address Line 1: Current practice street address (e.g., "123 Main Street")
    - Address Line 2: Suite/unit number (or empty string if none)
    - City: City name in ALL CAPS (e.g., "CHARLOTTE")
    - State: 2-letter US state code (e.g., TX, CA, NY)
    - ZIP: 5-digit zipcode (e.g., "28202")

    **Part 2 - Practice/Hospital Affiliation Details:**
    Search for the healthcare organization(s) where "{hcp_name}" practices.
    
    Search queries to try:
    - "{hcp_name} doctor hospital affiliation"
    - "{hcp_name} NPI {hcp_npi}" on npiregistry.cms.hhs.gov
    - "{hcp_name} practices at"
    - "{hcp_name} affiliated with"
    - "{hcp_name} hospital privileges"
    
    Look for terms like "practices at", "affiliated with", "hospital privileges", "medical staff at", "employed by" and such terms which convey the information .
    
    For each affiliated organization, you MUST provide:
    - NPI: The HCP (Health Care Provider)'s NPI number (10 digits). Use the provided {hcp_npi} if available, or search npiregistry.cms.hhs.gov
    - HCO_ID: The organization's NPI number (10 digits). Search "[organization name] NPI number" or check nppes.cms.hhs.gov. Use "N/A" only if truly not findable.
    - HCO_Name: Full name of the hospital, medical group, health system, or clinic where they practice (e.g., "Advocate Health", "HCA Healthcare", "CommonSpirit Health", "Ascension")
    - HCO_Address1: REQUIRED - Street address of the practice/hospital location. Search "[organization name] address" or check their website Contact page. Example: "3075 Highland Parkway"
    - HCO_City: REQUIRED - City in ALL CAPS. Example: "DOWNERS GROVE"
    - HCO_State: REQUIRED - 2-letter state code. Example: "IL"
    - HCO_ZIP: REQUIRED - 5-digit zipcode. Example: "60515"

    **CRITICAL:** 
    - If you find an affiliated organization name, you MUST search for "[organization name] address" and return the complete address.
    - Do NOT leave address fields as "N/A" if the organization exists - their address is publicly available.
    - Only return "N/A" for HCO fields if the provider truly has no known affiliations.
    - Return actual found data, not "N/A" unless truly not findable after thorough search.
    - Never return the name of the HCO as the HCO_Name - only return the actual organization name.
    """

    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": user_query}],
        web_search_options={
            "search_type": "pro" if use_pro_search else "fast"
        },  
        response_format={
            "type": "json_schema",
            "json_schema": {
                "schema": SearchResponse.model_json_schema()
            }
        }
    )

    return json.loads(completion.choices[0].message.content)

def standardize_value_lengths(dictionary):
    valid_lists = [v for v in dictionary.values() if isinstance(v, list) and len(v) > 0]
    if not valid_lists:
        return dictionary

    max_length = max(len(v) for v in valid_lists)

    for key, value in dictionary.items():
        if not isinstance(value, list):
            continue
        if len(value) == 0:
            dictionary[key] = [None] * max_length
        elif len(value) < max_length:
            dictionary[key].extend([value[0]] * (max_length - len(value)))

    return dictionary

                
# Corrected popup display logic
if st.session_state.current_view == "main":
    render_main_page(session)
elif st.session_state.current_view == "enrichment_page":
    # A placeholder container is created for the popup
    popup_placeholder = st.empty()
    if st.session_state.show_popup:
        show_popup_without_button(popup_placeholder, st.session_state.popup_message_info['type'], st.session_state.popup_message_info) 

    # Check if this is an empty record flow (from "Still want to proceed with Web Search?" button)
    if st.session_state.selected_hcp_id == 'empty_record' and st.session_state.get('empty_record_for_enrichment'):
        # Create a DataFrame from the empty record
        empty_record = st.session_state.empty_record_for_enrichment
        selected_record_df = pd.DataFrame([empty_record])
        
        if not st.session_state.show_popup:
            render_enrichment_page(session, selected_record_df)
    elif st.session_state.selected_hcp_id and st.session_state.results_df is not None:
        selected_record_df = st.session_state.results_df[
            st.session_state.results_df["ID"] == st.session_state.selected_hcp_id
        ]

        # This function call now needs to be wrapped in an `if` to prevent re-rendering issues
        # when a pop-up is active.
        if not st.session_state.show_popup:
            render_enrichment_page(session, selected_record_df)
    else:
        st.warning("Please select an HCP record from the main page first.")
        if st.button("Back to Main Page"):
            st.session_state.current_view = "main"
            st.rerun()
