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

    # --- LLM Data Enrichment Function ---
    # --- LLM Data Enrichment Function (UPDATED TO FIND ALL SOURCES) ---
    # --- LLM Data Enrichment Function (UPDATED TO EXTRACT URLS) ---
    @st.cache_data(ttl=600)
    def get_enriched_data_from_llm(_session, hcp_df):
        if hcp_df.empty:
            return pd.DataFrame()
    
        MODEL_NAME = "mistral-large"
        NUM_CHUNKS = 30
        CORTEX_SEARCH_DATABASE = "CORTEX_ANALYST_HCK"
        CORTEX_SEARCH_SCHEMA = "PUBLIC"
        CORTEX_SEARCH_SERVICE = "CC_SEARCH_SERVICE_CS"
        COLUMNS = ["chunk", "chunk_index", "relative_path", "category"]
    
        try:
            root = Root(_session)
            svc = (
                root.databases[CORTEX_SEARCH_DATABASE]
                .schemas[CORTEX_SEARCH_SCHEMA]
                .cortex_search_services[CORTEX_SEARCH_SERVICE]
            )
        except Exception as e:
            st.error(
                f"Could not connect to the Cortex Search Service for enrichment. Error: {e}"
            )
            return pd.DataFrame()
    
        selected_record = hcp_df.iloc[0].to_dict()
        search_query = f"{selected_record.get('NAME', '')} NPI {selected_record.get('NPI', '')}"
        # --- PROMPT UPDATED TO EXTRACT URLS AS SOURCE ---
        enrichment_prompt = f"""
        You are a data extraction assistant. Your only job is to read the document text provided  in the <context> tags and your own LLM training data information available,  and then find the exact values for the fields in the required JSON structure.
        Do not invent or infer information. If you cannot find a value for a specific field, return null.
    
        Your response MUST be ONLY a single, valid JSON object string.
        The JSON object must follow this exact structure:
        {{
            "ID": "{selected_record.get('ID', '')}", "Name": "...", "Name_Score": "...", "Name_Source": ["..."],
            "NPI": 0, "Address Line1": "...", "Address Line1_Score": "...", "Address Line1_Source": ["..."],
            "Address Line2": "...", "Address Line2_Score": "...", "Address Line2_Source": ["..."],
            "City": "...", "City_Score": "...", "City_Source": ["..."], "State": "...", "State_Score": "...", "State_Source": ["..."],
            "ZIP": "...", "ZIP_Score": "...", "ZIP_Source": ["..."],
    
            "HCO 1 ID": "...", "HCO 1 Name": "...", "HCO 1 NPI": "...", "HCO 1 Address Line1": "...", "HCO 1 Address Line2": "...", "HCO 1 City": "...", "HCO 1 State": "...", "HCO 1 ZIP": "...",
            "HCO 2 ID": "...", "HCO 2 Name": "...", "HCO 2 NPI": "...", "HCO 2 Address Line1": "...", "HCO 2 Address Line2": "...", "HCO 2 City": "...", "HCO 2 State": "...", "HCO 2 ZIP": "...",
            "HCO 3 ID": null, "HCO 3 Name": null, "HCO 3 NPI": null, "HCO 3 Address Line1": null, "HCO 3 Address Line2": null, "HCO 3 City": null, "HCO 3 State": null, "HCO 3 ZIP": null
        }}
    
        --- IMPORTANT RULES FOR SCORING & SOURCES ---
        - The context contains text from different documents about an HCP(Health Care Provider). Your task is to find the source URL (like `https://npiregistry.cms.hhs.gov/...`) within the text chunks you use.
        - For each proposed value, you MUST populate the corresponding *_Source field with a JSON array containing all full URLs you find that support the value.
        - If you cannot find a specific URL for a piece of data, return an array containing the document's 'category' as a fallback.
        - If no sources are found, return an empty array [].
        - The *_Score represents your confidence of proposed information being correct verified through multiple sources for the HCP. For eg: For an HCP given in context, if you are able to verify it's demographic information from multiple sources then score would be high compared to if the information is only fetched from one source. 
        -  Also in the *_Score fields, populate the confidence score in percentage(out of 100) followed by '%' and then followed by your reason for assigning that score to the respective field value proposed.
        """

        # Connect --> Connection string
    
        try:
            response = svc.search(search_query, COLUMNS, limit=NUM_CHUNKS)
            context_for_prompt = json.dumps(response.json())
            final_prompt_with_context = f"""
            You are an expert assistant. Extract information from the CONTEXT to answer the QUESTION.
            <context>{context_for_prompt}</context>
            <question>{enrichment_prompt}</question>
            """
            full_cmd = f"SELECT snowflake.cortex.complete('{MODEL_NAME}', $${final_prompt_with_context}$$) as response"
            #st.write(full_cmd)

            api_response = get_consolidated_data_for_hcp(selected_record, model_name="sonar-pro", use_pro_search=True)
            # hcp_data = standardize_value_lengths(api_response.hcp_data)
            # df_response = pd.DataFrame(hcp_data)
                                              
            # if df_response.empty:
            #     st.warning("The AI assistant returned an empty response.")
            #     return pd.DataFrame()
            # else:
            #     return df_response
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
    
    # Render confirmation dialog if the state is set
    if st.session_state.get('show_confirm_dialog'):
        with dialog_placeholder.container():
            st.warning("Are you sure you want to update the selected fields? This action cannot be undone.", icon="⚠️")
            
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
            
            if remaining_details_to_display:
                st.markdown("**Other profile details of the account (not changing):**")
                remaining_df = pd.DataFrame(remaining_details_to_display, columns=["Field", "Value"])
                st.dataframe(remaining_df, hide_index=True, use_container_width=True)

            st.markdown("---")
            # --- END MODIFIED ---
            
            # Use st.columns for horizontal buttons
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Yes, Update", key="confirm_yes"):
                    approved_df_cols = st.session_state.get('approved_cols', [])
                    selected_id = st.session_state.selected_hcp_id
                    
                    if not approved_df_cols:
                        st.info("No fields were selected for update. Please go back and select fields.")
                        st.session_state.show_confirm_dialog = False
                        st.rerun()
                    else:
                        with st.spinner("Updating record in Snowflake..."):
                            try:
                                db_column_map = {
                                    "Name": "NAME", "Address Line1": "ADDRESS1", "Address Line2": "ADDRESS2",
                                    "City": "CITY", "State": "STATE", "ZIP": "ZIP"
                                }
                                update_assignments = {}
                                proposed_record = st.session_state.proposed_record
                                updated_columns_list = []
                                for col_name in approved_df_cols:
                                    db_col_name = db_column_map.get(col_name)
                                    if db_col_name:
                                        new_value = proposed_record.get(col_name)
                                        if hasattr(new_value, 'item'): new_value = new_value.item()
                                        update_assignments[db_col_name] = new_value
                                        updated_columns_list.append(db_col_name)

                                DATABASE, SCHEMA, YOUR_TABLE_NAME = "CORTEX_ANALYST_HCK", "PUBLIC", "NPI"
                                target_table = session.table(f'"{DATABASE}"."{SCHEMA}"."{YOUR_TABLE_NAME}"')
                                update_result = target_table.update(update_assignments, col("ID") == selected_id)

                                if update_result.rows_updated > 0:
                                    updated_cols_str = ", ".join(updated_columns_list)
                                    custom_message = f"Record for ID: {selected_id} updated successfully. Changed columns: {updated_cols_str}."
                                    st.session_state.show_popup = True
                                    st.session_state.popup_message_info = { 'type': 'update_success', 'id': selected_id, 'message': custom_message }
                                else:
                                    st.warning(f"Record for ID {selected_id} was not found for update.")
                                    st.session_state.show_confirm_dialog = False
                                    st.rerun()
                            except Exception as e:
                                st.error(f"An error occurred while updating the record: {e}")
                                st.session_state.show_confirm_dialog = False
                                st.rerun()
                                
                            st.session_state.show_confirm_dialog = False
                            st.rerun()
            with col2:
                if st.button("Cancel", key="confirm_cancel"):
                    st.session_state.show_confirm_dialog = False
                    st.rerun()
        return
    
    # Check for primary update confirmation dialog
    if st.session_state.get('show_primary_confirm_dialog'):
        with dialog_placeholder.container():
            st.warning("Are you sure you want to change the primary affiliation? This will update the main record.", icon="⚠️")
            
            # --- MODIFIED: Display primary affiliation change in a vertical table ---
            current_primary_id = selected_record.get("PRIMARY_AFFL_HCO_ACCOUNT_ID")
            current_primary_name_query = session.sql(f"SELECT HCO_NAME FROM HCP_HCO_AFFILIATION WHERE HCO_ID = '{current_primary_id}'").collect() if current_primary_id else None
            current_primary_name = current_primary_name_query[0].HCO_NAME if current_primary_name_query and not current_primary_name_query[0].HCO_NAME is None else "N/A"
            
            new_primary_id = st.session_state.primary_hco_id
            new_primary_name_query = session.sql(f"SELECT HCO_NAME FROM HCP_HCO_AFFILIATION WHERE HCO_ID = '{new_primary_id}'").collect() if new_primary_id else None
            new_primary_name = new_primary_name_query[0].HCO_NAME if new_primary_name_query and not new_primary_name_query[0].HCO_NAME is None else "N/A"

            primary_change_df = pd.DataFrame({
                "Field": ["ID", "Name", "Current Primary HCO", "Proposed Primary HCO"],
                "Value": [
                    selected_record.get('ID'),
                    selected_record.get('NAME'),
                    f"ID: {current_primary_id} ({current_primary_name})",
                    f"ID: {new_primary_id} ({new_primary_name})"
                ]
            })
            st.dataframe(primary_change_df.set_index('Field'), use_container_width=True)
            # --- END MODIFIED ---

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Yes, Set Primary", key="confirm_primary_yes"):
                    new_primary_id = st.session_state.primary_hco_id
                    selected_id = st.session_state.selected_hcp_id
                    
                    with st.spinner("Updating primary affiliation in Snowflake..."):
                        try:
                            npi_table = session.table("NPI")
                            update_assignments = {"PRIMARY_AFFL_HCO_ACCOUNT_ID": new_primary_id}
                            update_result = npi_table.update(update_assignments, col("ID") == selected_id)
                            if update_result.rows_updated > 0:
                                st.session_state.show_popup = True
                                st.session_state.popup_message_info = { 'type': 'primary_success', 'hco_id': new_primary_id }
                                st.session_state.show_primary_confirm_dialog = False
                                st.rerun()
                            else:
                                st.warning("Could not find the main HCP record to update.")
                                st.session_state.show_primary_confirm_dialog = False
                                st.rerun()
                        except Exception as e:
                            st.error(f"An error occurred during the update: {e}")
                            st.session_state.show_primary_confirm_dialog = False
                            st.rerun()
            with col2:
                if st.button("Cancel", key="confirm_primary_cancel"):
                    st.session_state.show_primary_confirm_dialog = False
                    st.rerun()
        return
#end of Placeholder for a potential dialog to display over the main content

    with st.spinner("🚀 Contacting AI Assistant for Data Enrichment..."):
        # proposed_df = get_enriched_data_from_llm(session, selected_hcp_df)
        api_response = get_enriched_data_from_llm(session, selected_hcp_df)
        proposed_hcp_data_df = pd.DataFrame(api_response['hcp_data'])
        proposed_hcp_affiliation_data_df = pd.DataFrame(api_response['hcp_affiliation_data'])

    try:
        if current_df.empty or proposed_hcp_data_df.empty:
            st.warning("Could not generate a comparison report.")
            st.stop()
    except AttributeError:
        st.error("One of the dataframes is invalid. Please check the data source.")
        st.stop()

    selected_id = int(current_df['ID'].iloc[0])
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
        with btn_col:
            if st.button("Update Record 💾", type="primary", key=f"update_btn_{selected_id}"):
                approved_df_cols = []
                for field_label, col_name in provider_mapping.items():
                    checkbox_key = f"approve_{selected_id}_{col_name}"
                    if st.session_state.get(checkbox_key, False):
                        approved_df_cols.append(col_name)

                if approved_df_cols:
                    st.session_state.show_confirm_dialog = True
                    st.session_state.approved_cols = approved_df_cols
                    st.rerun()
                else:
                    st.info(f"No fields were selected for update for ID {selected_id}.")

    st.markdown("<hr style='margin-top: 0; margin-bottom: 0; border-top: 1px solid #ccc;'>", unsafe_allow_html=True)
    
    hco_affiliation_title = f"HCO Affiliation information of : {current_record.get('Name', 'N/A')} (NPI: {current_record.get('NPI', 'N/A')})"
    
    with st.expander(hco_affiliation_title, expanded=False):
        
        hco_headers = ["Status", "SOURCE", "HCP NPI", "HCO ID", "HCO NAME", "HCO ADDRESS", "HCO CITY", "HCO STATE", "HCO ZIP"]
        header_cols = st.columns([1.5, 2, 1.5, 1.5, 2.5, 2, 1.5, 1.5, 1.5])
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
        # st.write("DEBUG - proposed_hcp_affiliation_data_df columns:", proposed_hcp_affiliation_data_df.columns.tolist())
        # st.write("DEBUG - proposed_hcp_affiliation_data_df:", proposed_hcp_affiliation_data_df)
        if not proposed_hcp_affiliation_data_df.empty:
            for index, row in proposed_hcp_affiliation_data_df.iterrows():
                hco_name = row.get('HCO_Name')
                if pd.notna(hco_name) and str(hco_name).strip() != "":
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
            sorted_affiliations = sorted(
                all_affiliations.items(),
                key=lambda item: (
                    pd.notna(item[0]) and
                    true_primary_hco_id is not None and
                    int(item[0]) == true_primary_hco_id
                ),
                reverse=True
            )
            
            for hco_id, hco_data in sorted_affiliations:
                row_cols = st.columns([1.5, 2, 1.5, 1.5, 2.5, 2, 1.5, 1.5, 1.5])
                
                is_primary = pd.notna(hco_id) and true_primary_hco_id is not None and int(hco_id) == true_primary_hco_id
                
                with row_cols[0]:
                    if is_primary:
                        st.markdown("✅ **Primary**")
                    else:
                        if st.button("Set as Primary", key=f"set_primary_{hco_id}"):
                            st.session_state.show_primary_confirm_dialog = True
                            st.session_state.primary_hco_id = hco_id
                            st.rerun()
                
                source = hco_data.get("SOURCE", "")
                is_ai_source = (source == "Generated by AI")
                row_cols[1].write(source)
                
                if is_ai_source:
                    row_cols[2].write("")
                else:
                    hcp_npi_val = hco_data.get("HCP_NPI")
                    row_cols[2].write(str(hcp_npi_val) if pd.notna(hcp_npi_val) else "")
                    
                if is_ai_source:
                    row_cols[3].write("")
                else:
                    row_cols[3].write(str(hco_data.get("HCO ID", "")))
        
                row_cols[4].write(hco_data.get("HCO NAME", ""))
                row_cols[5].write(hco_data.get("HCO ADDRESS", ""))
                row_cols[6].write(hco_data.get("HCO CITY", ""))
                row_cols[7].write(hco_data.get("HCO STATE", ""))
                row_cols[8].write(hco_data.get("HCO ZIP", ""))

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
        if not sql_item_found:
            st.info("The assistant did not return a SQL query for this prompt. It may be a greeting or a clarifying question.")

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
                            
                            primary_hco_id = selected_record.get("PRIMARY_AFFL_HCO_ACCOUNT_ID")
                            
                            hco_col1.markdown(f'<div class="detail-key">HCP ID:</div><div class="detail-value">{get_safe_value(selected_record, "ID")}</div>', unsafe_allow_html=True)
                            
                            hco_id_val = str(int(primary_hco_id)) if pd.notna(primary_hco_id) and primary_hco_id is not None else "N/A"
                            hco_col2.markdown(f'<div class="detail-key">Primary HCO NPI:</div><div class="detail-value">{hco_id_val}</div>', unsafe_allow_html=True)
                            
                            hco_col1.markdown(f'<div class="detail-key">HCP Name:</div><div class="detail-value">{get_safe_value(selected_record, "NAME")}</div>', unsafe_allow_html=True)
                            # Removed the line for "Primary HCO Name" as requested.
                            
                            
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
    

def get_consolidated_data_for_hcp(hcp_data, model_name="sonar", use_pro_search=False):
    # Extract key info for better search - handle both dict and pandas Series
    if hasattr(hcp_data, 'to_dict'):
        hcp_data = hcp_data.to_dict()
    hcp_name = hcp_data.get('NAME', '') if isinstance(hcp_data, dict) else str(hcp_data)
    hcp_npi = hcp_data.get('NPI', '') if isinstance(hcp_data, dict) else ''
    
    user_query = f"""
    Search the web for information about this US healthcare provider:
    
    Name: {hcp_name}
    NPI: {hcp_npi}

    Find and return:

    **Part 1 - Provider Demographics (verify/update from web sources):**
    - Name: Full name of the doctor/provider
    - Address Line 1: Current practice street address
    - Address Line 2: Suite/unit number (or empty string if none)
    - City: City name in ALL CAPS
    - State: 2-letter US state code (e.g., TX, CA, NY)
    - ZIP: 5-digit zipcode

    **Part 2 - Practice/Hospital Affiliation:**
    Search NPI Registry, hospital websites, Healthgrades, Vitals, WebMD, or Doximity for where this doctor practices.
    - NPI: The HCP (Health Care Provider)'s NPI number (10 digits) - search npiregistry.cms.hhs.gov
    - HCO_ID: Use the organization NPI as ID, or "N/A" if not found
    - HCO_Name: Name of the hospital, medical group, or clinic where they practice
    - HCO_Address1: Street address of the practice location
    - HCO_City: City in ALL CAPS
    - HCO_State: 2-letter state code
    - HCO_ZIP: 5-digit zipcode

    **Search Tips:**
    - Search "{hcp_name} doctor hospital affiliation"
    - Search "{hcp_name} NPI {hcp_npi}" on npiregistry.cms.hhs.gov
    - Look for "practices at", "affiliated with", "hospital privileges"
    - Return actual found data, not "N/A" unless truly not findable
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

    if st.session_state.selected_hcp_id and st.session_state.results_df is not None:
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
