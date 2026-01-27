import json
import time
import os
import requests
import pandas as pd
import streamlit as st
from snowflake.core import Root
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col
from snowflake.cortex import Complete
from urllib.parse import urlparse
from typing import List, Optional
from pydantic import BaseModel, Field
from perplexity import Perplexity

# Utility Functions
def get_perplexity_client():
    os.environ["PERPLEXITY_API_KEY"] = st.secrets["perplexity"]["api_key"]
    return Perplexity()

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

# Cortex LLM Priority Fetching Functions
def get_hcp_affiliation_priorities_from_llm(session, selected_hcp_data: dict, affiliations: list) -> dict:
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

def get_hco_affiliation_priorities_from_llm(session, selected_hco_data: dict, affiliations: list) -> dict:
    """
    Calls Snowflake Cortex REST API with structured JSON output to rank HCO affiliations by priority.
    
    Returns a dict mapping affiliation key to {"priority": int, "reason": str}
    """
    if not affiliations:
        return {}
    
    # Build the prompt for the LLM
    selected_info = f"""
Selected Healthcare Organization:
- Name: {selected_hco_data.get('Name', 'N/A')}
- Address: {selected_hco_data.get('Address Line1', '')} {selected_hco_data.get('Address Line2', '')}
- City: {selected_hco_data.get('City', 'N/A')}
- State: {selected_hco_data.get('State', 'N/A')}
- ZIP: {selected_hco_data.get('ZIP', 'N/A')}
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
    
    prompt = f"""You are a healthcare data analyst. Analyze the following selected healthcare organization and its potential affiliations. 
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

# Pages Rendering
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
    FILE = "HCO_MODEL.yaml"

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
        # Clear previous messages and results
        st.session_state.messages.clear()
        st.session_state.results_df = None

        st.session_state.provider_info_change = False

        # Clear previous hcp_id selection
        st.session_state.selected_hco_id = None

        # Add user message to session state
        st.session_state.messages.append(
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        )
        with st.spinner("Generating response..."):
            try:
                response = send_message(prompt=prompt)
                question_item = {"type": "text", "text": prompt.strip()}
                response["message"]["content"].insert(0, question_item)

                # Add assistant message to session state
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

    def ensure_join_in_sql(sql: str) -> str:
        """
        Ensures the SQL always includes a LEFT JOIN with OUTLET_HCO_AFFILIATION.
        If Cortex only queries HCO table, wrap it to include the join.
        """
        sql_upper = sql.upper()
        # Check if join is already present
        if "OUTLET_HCO_AFFILIATION" in sql_upper or "LEFT OUTER JOIN" in sql_upper or "LEFT JOIN" in sql_upper:
            return sql
        
        # Extract WHERE clause if present
        where_clause = ""
        if " WHERE " in sql_upper:
            where_idx = sql_upper.index(" WHERE ")
            where_clause = sql[where_idx:]
            # Remove ORDER BY from where_clause for re-adding later
            if " ORDER BY " in where_clause.upper():
                order_idx = where_clause.upper().index(" ORDER BY ")
                where_clause = where_clause[:order_idx]
        
        # Extract the name filter from WHERE clause
        name_filter = ""
        if "NAME ILIKE" in sql_upper:
            import re
            match = re.search(r"NAME\s+ILIKE\s+'([^']+)'", sql, re.IGNORECASE)
            if match:
                name_filter = match.group(1)
        
        if name_filter:
            # Build a new query with join but use subquery to get only first affiliation per HCO
            new_sql = f"""
            SELECT h.ID, h.NAME, h.ADDRESS1, h.ADDRESS2, h.CITY, h.STATE, h.ZIP, h.COUNTRY,
                   o.OUTLET_ID, o.OUTLET_NAME, o.OUTLET_ADDRESS1, o.OUTLET_CITY, o.OUTLET_STATE, o.OUTLET_ZIP
            FROM CORTEX_ANALYST_HCK.PUBLIC.HCO h
            LEFT OUTER JOIN (
                SELECT HCO_ID, OUTLET_ID, OUTLET_NAME, OUTLET_ADDRESS1, OUTLET_CITY, OUTLET_STATE, OUTLET_ZIP,
                       ROW_NUMBER() OVER (PARTITION BY HCO_ID ORDER BY OUTLET_ID) as rn
                FROM CORTEX_ANALYST_HCK.PUBLIC.OUTLET_HCO_AFFILIATION
            ) o ON h.ID = o.HCO_ID AND o.rn = 1
            WHERE h.NAME ILIKE '{name_filter}'
            ORDER BY h.NAME
            """
            return new_sql.strip()
        
        return sql

    def display_results_table(content: list):
        sql_item_found = False
        for item in content:
            if item["type"] == "sql":
                sql_item_found = True
                with st.spinner("Running SQL..."):
                    original_sql = item["statement"]
                    sql_to_run = ensure_join_in_sql(original_sql)
                    df = session.sql(sql_to_run).to_pandas()
                    if not df.empty:
                        st.session_state.results_df = df
                        st.write("Please select a record from the table to proceed:")
                        
                        # Define column sizes tuple (must match number of headers)
                        col_sizes = (0.8, 0.8, 2, 2, 1.5, 1)

                        # Define column heading names
                        cols = st.columns(col_sizes)
                        headers = ["Select", "ID", "Name", "Address", "City", "State"]
                        
                        # Render table headers
                        for col_header, header_name in zip(cols, headers):
                            col_header.markdown(f"**{header_name}**")

                        # Render table rows
                        for index, row in df.iterrows():
                            # Check for ID column with various possible names from Cortex
                            row_id = row.get("ID") if "ID" in row.index else row.get("HCO_ID")
                            if row_id is None or pd.isna(row_id):
                                row_id = index  # Fallback to index if ID is NULL
                            is_selected = row_id == st.session_state.get("selected_hco_id")
                            row_cols = st.columns(col_sizes)

                            if is_selected:
                                row_cols[0].write("🔘")
                            else:
                                # Use index in key to ensure uniqueness even if row_id duplicates
                                if row_cols[0].button("", key=f"select_{row_id}_{index}"):
                                    st.session_state.selected_hco_id = row_id
                                    st.rerun()
                            row_cols[1].write(row_id)
                            row_cols[2].write(row.get("NAME") or row.get("HCO_NAME", ""))
                            row_cols[3].write(row.get("ADDRESS1") or row.get("HCO_ADDRESS1", "N/A"))
                            row_cols[4].write(row.get("CITY") or row.get("HCO_CITY", "N/A"))
                            row_cols[5].write(row.get("STATE") or row.get("HCO_STATE", "N/A"))
                    else:
                        st.info("We couldn't find any records matching your search.", icon="ℹ️")
                        st.markdown("")
                        if st.button("🔍 Still want to proceed with Web Search?", type="primary"):
                            # Create a default empty record for enrichment
                            st.session_state.empty_record_for_enrichment = {
                                'ID': 'N/A',
                                'NAME': st.session_state.get('web_search_query', '').title().strip(),
                                'NPI': '',
                                'ADDRESS1': '',
                                'ADDRESS2': '',
                                'CITY': '',
                                'STATE': '',
                                'ZIP': '',
                                'COUNTRY': '',
                                'PRIMARY_AFFL_HCO_ACCOUNT_ID': None,
                                'OUTLET_ID': None,
                                'OUTLET_NAME': '',
                                'OUTLET_ADDRESS1': '',
                                'OUTLET_CITY': '',
                                'OUTLET_STATE': '',
                                'OUTLET_ZIP': ''
                            }
                            # Store the search query for web search context
                            st.session_state.web_search_query = st.session_state.get('last_prompt', '')
                            st.session_state.selected_hco_id = 'empty_record'
                            st.session_state.current_view = "enrichment_page"
                            st.rerun()
        if not sql_item_found:
            st.info("The assistant did not return a SQL query for this prompt. It may be a greeting or a clarifying question.")
            st.markdown("")
            if st.button("🔍 Still want to proceed with Web Search?", key="web_search_no_sql", type="primary"):
                # Create a default empty record for enrichment
                st.session_state.empty_record_for_enrichment = {
                    'ID': 'N/A',
                    'NAME': None,
                    'ADDRESS1': None,
                    'ADDRESS2': None,
                    'CITY': None,
                    'STATE': None,
                    'ZIP': None,
                    'COUNTRY': None
                }
                # Store the search query for web search context
                st.session_state.web_search_query = st.session_state.get('last_prompt', '')
                st.session_state.selected_hco_id = 'empty_record'
                st.session_state.current_view = "enrichment_page"
                st.rerun()

    # Select App Type
    option = st.selectbox(
        "Select App Type",
        ("HCP Data Steward", "HCO Data Steward"),
        index=None,
        placeholder="Select app type...",
    )
    
    # Set app_variant based on selection
    if option == "HCP Data Steward":
        st.session_state.app_variant = "HCP"
    elif option == "HCO Data Steward":
        st.session_state.app_variant = "HCO"

    st.write(f"DEBUG: Selected App Variant - {st.session_state.app_variant}")

    if st.session_state.app_variant is not None:
        pass
        # # Search Input (HCP/HCO) (based on app_variant)
        # freeze_container = st.container(border=True)
        # with freeze_container:
        #     user_input_text = st.chat_input(f"Search for an {st.session_state.app_variant} Account")
        #     current_prompt = user_input_text

        #     if current_prompt and current_prompt != st.session_state.get("last_prompt"):
        #         process_message(prompt=current_prompt)
        #         st.session_state.last_prompt = current_prompt

        # # --- DISPLAY LOGIC (Vertical Flow) ---
        # if st.session_state.messages:
        # assistant_messages = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
        # if assistant_messages:
        #     st.markdown("---")
        #     display_interpretation(content=assistant_messages[-1]["content"])

        #     # 1. Search Results Table (Full Width)
        #     response_container = st.container(border=True)
        #     with response_container:
        #         st.subheader("Search Results")
        #         display_results_table(content=assistant_messages[-1]["content"])

        #     # Helper to safely get and format value (handles both column naming conventions)
        #     def get_safe_value(record, key):
        #         # Try original key first, then with HCO_ prefix
        #         value = record.get(key)
        #         if value is None or (isinstance(value, float) and pd.isna(value)):
        #             value = record.get(f"HCO_{key}")
        #         return str(value) if pd.notna(value) and value is not None else 'N/A'
            

        #     # 2. Selected Record Details (Only appears when selected_hco_id is set)
        #     if st.session_state.get("selected_hco_id") and st.session_state.get("results_df") is not None:
        #         # Handle both ID and HCO_ID column names
        #         id_col = "ID" if "ID" in st.session_state.results_df.columns else "HCO_ID"
        #         # Convert both to string for comparison to avoid type mismatch
        #         selected_id_str = str(st.session_state.selected_hco_id)
        #         selected_record_df = st.session_state.results_df[
        #             st.session_state.results_df[id_col].astype(str) == selected_id_str
        #         ]

                
        #         if not selected_record_df.empty:
                    
        #             selected_record = selected_record_df.iloc[0]
                    
        #             # --- Start Two-Column Layout for Details Sections (Side-by-Side Below Search) ---
        #             details_col_left, details_col_right = st.columns(2)
                    
        #             # --- Left Detail Column: Current Demographic Details ---
        #             with details_col_left:
        #                 st.subheader("Current HCO Address Details")
                        
        #                 # This border container holds the custom 2x2 layout
        #                 with st.container(border=True):
                            
        #                     # ID and Name in the required structure (single line)
        #                     hcp_id = get_safe_value(selected_record, 'ID') if 'ID' in selected_record.index else get_safe_value(selected_record, 'HCO_ID')
        #                     hcp_name = get_safe_value(selected_record, 'NAME') if 'NAME' in selected_record.index else get_safe_value(selected_record, 'HCO_NAME')
        #                     st.markdown(f'**ID:** {hcp_id} - {hcp_name}', unsafe_allow_html=True)
                            
        #                     st.markdown("<hr style='margin-top: 0; margin-bottom: 0; border-top: 1px solid #ccc;'>", unsafe_allow_html=True)

        #                     # Define the fields for the new two-column layout
        #                     left_fields = [                                
        #                         ("Address Line 1", "ADDRESS1"),
        #                         ("Address Line 2", "ADDRESS2"),
        #                         ("City", "CITY"),]
        #                     right_fields = [
        #                         ("State", "STATE"),
        #                         ("ZIP", "ZIP"),
        #                         ("Country", "COUNTRY")
        #                     ]

        #                     # Create two internal columns for the key-value pairs
        #                     left_col_address, right_col_address = st.columns(2)

        #                     # Render Identity Fields (Left Column)
        #                     for label, key in left_fields:
        #                         value = get_safe_value(selected_record, key)
        #                         display_value = value if value and value.strip() else 'N/A'
        #                         left_col_address.markdown(
        #                             f'<div class="detail-key">{label}:</div>'
        #                             f'<div class="detail-value">{display_value}</div>',
        #                             unsafe_allow_html=True
        #                         )

        #                     # Render Address Fields
        #                     for label, key in right_fields:
        #                         value = get_safe_value(selected_record, key)
        #                         display_value = value if value and value.strip() else 'N/A'
        #                         right_col_address.markdown(
        #                             f'<div class="detail-key">{label}:</div>'
        #                             f'<div class="detail-value">{display_value}</div>',
        #                             unsafe_allow_html=True
        #                         )

        #             # --- Right Detail Column: Primary HCO Affiliation Details ---
        #             with details_col_right:
        #                 st.subheader("Primary HCO Affiliation Details")
        #                 with st.container(border=True):
        #                     hco_col1, hco_col2 = st.columns(2)
        #                     primary_hco_id = selected_record.get("OUTLET_ID")
                            
        #                     hco_col1.markdown(f'<div class="detail-key">Parent ID:</div><div class="detail-value">{get_safe_value(selected_record, "OUTLET_ID")}</div>', unsafe_allow_html=True)
                            
        #                     hco_id_val = str(int(primary_hco_id)) if pd.notna(primary_hco_id) and primary_hco_id is not None else "N/A"

        #                     hco_col2.markdown(f'<div class="detail-key">Parent HCO NPI:</div><div class="detail-value">{hco_id_val}</div>', unsafe_allow_html=True)
                            
        #                     hco_col1.markdown(f'<div class="detail-key">Parent Name:</div><div class="detail-value">{get_safe_value(selected_record, "OUTLET_NAME")}</div>', unsafe_allow_html=True)
        #                     # Removed the line for "Primary HCO Name" as requested.
                            
                            
        #             # --- End Two-Column Layout for Details Sections ---
        #             st.divider()
                    
        #             # --- Enrich Button (Full Width, below the detail columns) ---
        #             # -----------------------------------------------------------
        #             # MODIFIED SECTION STARTS HERE
        #             # -----------------------------------------------------------
        #             # Create two columns, with a small width for the button
        #             button_col, _ = st.columns([0.2, 0.8])
                    
        #             with button_col:
        #                 if st.button("Enrich with AI Assistant 🚀", type="primary"):
        #                     st.session_state.current_view = "enrichment_page"
        #                     st.rerun()
        #             # -----------------------------------------------------------
        #             # MODIFIED SECTION ENDS HERE
        #             # -----------------------------------------------------------
                    
        #         else:
        #             st.info("No data found for selected record ID.")
    


# Initialize session state variables
if "app_variant" not in st.session_state:
    st.session_state.app_variant = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "selected_hco_id" not in st.session_state:
    st.session_state.selected_hco_id = None
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
if "show_reason_popup" not in st.session_state:
    st.session_state.show_reason_popup = False
if "reason_popup_data" not in st.session_state:
    st.session_state.reason_popup_data = None
if "priority_reasons" not in st.session_state:
    st.session_state.priority_reasons = {}
if "selected_hcp_id" not in st.session_state:
    st.session_state.selected_hcp_id = None
if "primary_hco_data" not in st.session_state:
    st.session_state.primary_hco_data = None

session = get_snowflake_session()

# Page Router
if st.session_state.current_view == "main":
    render_main_page(session)
