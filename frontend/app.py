import streamlit as st
import pandas as pd
import io
import math
import requests
from typing import Optional, Tuple

# Configure page
st.set_page_config(
    page_title="Excel Data Viewer",
    page_icon="📊",
    layout="wide"
)

def load_excel_file(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load Excel file and return DataFrame
    
    Args:
        uploaded_file: Streamlit uploaded file object
    Returns:
        DataFrame if successful, None if error
    """
    try:
        # Determine file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension in ['xlsx', 'xls']:
            # Read Excel file
            df = pd.read_excel(uploaded_file)
            return df
        else:
            st.error("Please upload a valid Excel file (.xlsx or .xls)")
            return None
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return None

def display_pagination_controls(total_rows: int, rows_per_page: int, current_page: int, position: str = "top") -> int:
    """
    Display pagination controls and return selected page
    
    Args:
        total_rows: Total number of rows in dataset
        rows_per_page: Number of rows per page
        current_page: Current page number (0-indexed)
        position: "top" or "bottom" to distinguish button keys
        
    Returns:
        Selected page number (0-indexed)
    """
    total_pages = math.ceil(total_rows / rows_per_page)
    
    if total_pages <= 1:
        return 0
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col3:
        # Page selector
        page_options = list(range(1, total_pages + 1))
        selected_page_display = st.selectbox(
            f"Page (of {total_pages})",
            page_options,
            index=current_page,
            key=f"page_selector_{position}"
        )
        return selected_page_display - 1  # Convert to 0-indexed
    
    with col4:
        if st.button("Last", disabled=current_page == total_pages - 1, key=f"last_{position}"):
            return total_pages - 1
    
    with col5:
        if st.button("Next ➡️", disabled=current_page == total_pages - 1, key=f"next_{position}"):
            return min(total_pages - 1, current_page + 1)
    
    return current_page

def display_data_info(df: pd.DataFrame, current_page: int, rows_per_page: int):
    """
    Display information about the dataset and current view
    
    Args:
        df: DataFrame to display info for
        current_page: Current page number (0-indexed)
        rows_per_page: Number of rows per page
    """
    total_rows = len(df)
    start_row = current_page * rows_per_page + 1
    end_row = min((current_page + 1) * rows_per_page, total_rows)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", f"{total_rows:,}")
    
    with col2:
        st.metric("Total Columns", len(df.columns))
    
    with col3:
        st.metric("Current View", f"{start_row:,} - {end_row:,}")

def main():
    """Main application function"""
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False
    if 'show_config' not in st.session_state:
        st.session_state.show_config = False
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    
    # Header
    st.title("📊 Excel Data Viewer")
    st.markdown("Upload Excel files and view data with pagination for large datasets")
    
    # Sidebar with features menu
    with st.sidebar:
        # Header section
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <div style="background-color: #4285f4; color: white; padding: 10px; border-radius: 8px; margin-right: 10px;">
                📊
            </div>
            <div>
                <h3 style="margin: 0; color: #333;">Excel Data Viewer</h3>
                <p style="margin: 0; color: #666; font-size: 14px;">Data Processing</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Features section
        st.markdown("<p style='color: #999; font-size: 12px; margin-bottom: 10px;'>FEATURES</p>", unsafe_allow_html=True)
        
        # File Upload feature (highlighted)
        st.markdown("""
        <div style="background-color: #e8f0fe; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #4285f4;">
            <div style="display: flex; align-items: center;">
                <span style="margin-right: 8px;">📁</span>
                <span style="color: #4285f4; font-weight: 500;">File Upload</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Configuration feature
        if st.button("⚙️ Configuration", width="stretch", key="config_btn"):
            st.session_state.show_config = not st.session_state.get('show_config', False)
        
        # Data Analysis feature  
        st.markdown("""
        <div style="padding: 12px; border-radius: 8px; margin-bottom: 8px; cursor: pointer;">
            <div style="display: flex; align-items: center;">
                <span style="margin-right: 8px;">📊</span>
                <span style="color: #666;">Data Analysis</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show configuration if toggled
        if st.session_state.get('show_config', False):
            st.divider()
            st.subheader("Settings")
            
            # Rows per page selector
            rows_per_page = st.selectbox(
                "Rows per page",
                options=[10, 25, 50, 100, 200],
                index=2,  # Default to 50
                key="rows_per_page"
            )
        else:
            rows_per_page = 50  # Default value
        
        # Dataset summary if data exists
        if st.session_state.data is not None:
            st.divider()
            st.subheader("📈 Dataset Summary")
            st.write(f"**Shape:** {st.session_state.data.shape[0]:,} rows × {st.session_state.data.shape[1]} columns")
            st.write(f"**Memory Usage:** {st.session_state.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Processing metrics if available
        if st.session_state.get('metrics') is not None and st.session_state.file_processed:
            st.divider()
            st.subheader("📊 Processing Metrics")
            metrics = st.session_state.metrics
            
            # MAE First Date
            mae_first = metrics.get('mae_first_date', 'N/A')
            if mae_first != 'N/A':
                st.metric("MAE First Date", mae_first)
            
            # MAE Final Date  
            mae_final = metrics.get('mae_final_date', 'N/A')
            if mae_final != 'N/A':
                st.metric("MAE Final Date", mae_final)
            
            # IoU Score
            iou_value = metrics.get('iou', 'N/A')
            if iou_value != 'N/A':
                if isinstance(iou_value, (int, float)):
                    st.metric("IoU Score", f"{iou_value:.4f}")
                else:
                    st.metric("IoU Score", str(iou_value))
    
    # File upload section
    st.header("📁 Multiple File Upload")
    st.markdown("Upload 3 Excel files in the following order:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. MDL History")
        mdl_history_file = st.file_uploader(
            "Choose MDL History file",
            type=['xlsx', 'xls'],
            help="Upload MDL History Excel file",
            key="mdl_history"
        )
    
    with col2:
        st.subheader("2. MDL Input")
        mdl_input_file = st.file_uploader(
            "Choose MDL Input file",
            type=['xlsx', 'xls'],
            help="Upload MDL Input Excel file",
            key="mdl_input"
        )
    
    with col3:
        st.subheader("3. L3 File")
        l3_file = st.file_uploader(
            "Choose L3 file",
            type=['xlsx', 'xls'],
            help="Upload L3 Excel file",
            key="l3_file"
        )
    
    # Show upload status
    files_uploaded = [mdl_history_file, mdl_input_file, l3_file]
    file_names = ["MDL History", "MDL Input", "L3 File"]
    
    st.markdown("### Upload Status")
    for i, (file, name) in enumerate(zip(files_uploaded, file_names)):
        if file:
            st.success(f"✅ {name}: {file.name}")
        else:
            st.warning(f"⏳ {name}: Not uploaded yet")
    
    # RUN button and processing
    all_files_uploaded = all(files_uploaded)
    
    if all_files_uploaded:
        col1, col2 = st.columns([1, 4])
        
        with col1:
            run_button = st.button("🚀 PROCESS ALL FILES", type="primary", width="stretch")
        
        with col2:
            if not st.session_state.file_processed:
                st.info("Click 'PROCESS ALL FILES' to process the uploaded files in order")
            else:
                st.success("Files processed successfully!")
        
        # Process files when RUN button is clicked
        if run_button:
            with st.spinner("Processing all Excel files..."):
                # Reset pagination when new files are processed
                st.session_state.current_page = 0
                
                # Send files to backend API
                try:
                    # Prepare files for API request
                    files = {
                        "mdl_history": (mdl_history_file.name, mdl_history_file.getvalue(), mdl_history_file.type),
                        "mdl_input": (mdl_input_file.name, mdl_input_file.getvalue(), mdl_input_file.type),
                        "l3_file": (l3_file.name, l3_file.getvalue(), l3_file.type)
                    }
                    
                    # Make API request to backend
                    response = requests.post("http://0.0.0.0:3005/v1/upload-multiple-excel", files=files)
                    
                    if response.status_code == 200:
                        # Backend processing successful
                        backend_data = response.json()
                        st.success("✅ All files uploaded and processed by backend!")
                        
                        # Display backend response info
                        if 'info' in backend_data:
                            info = backend_data['info']
                            st.info(f"Backend processed {info.get('total_rows_processed', 0)} rows from {len(info.get('processed_files', []))} files")
                            
                            # Show file processing order
                            if 'file_order' in info:
                                st.success(f"✅ Files processed in order: {' → '.join(info['file_order'])}")
                            
                            # Display processing metrics in a nice layout
                            if any(key in info for key in ['mae_first_date', 'mae_final_date', 'iou']):
                                st.markdown("### 📊 Processing Metrics")
                                
                                metric_col1, metric_col2, metric_col3 = st.columns(3)
                                
                                with metric_col1:
                                    mae_first = info.get('mae_first_date', 'N/A')
                                    st.metric(
                                        label="MAE First Date",
                                        value=mae_first if mae_first != 'N/A' else 'Not Available',
                                        help="Mean Absolute Error for the first date"
                                    )
                                
                                with metric_col2:
                                    mae_final = info.get('mae_final_date', 'N/A')
                                    st.metric(
                                        label="MAE Final Date", 
                                        value=mae_final if mae_final != 'N/A' else 'Not Available',
                                        help="Mean Absolute Error for the final date"
                                    )
                                
                                with metric_col3:
                                    iou_value = info.get('iou', 'N/A')
                                    # Format IOU value if it's a number
                                    if isinstance(iou_value, (int, float)):
                                        iou_display = f"{iou_value:.4f}"
                                    else:
                                        iou_display = str(iou_value) if iou_value != 'N/A' else 'Not Available'
                                    
                                    st.metric(
                                        label="IoU Score",
                                        value=iou_display,
                                        help="Intersection over Union score"
                                    )
                                
                                # Store metrics in session state for later display
                                st.session_state.metrics = {
                                    'mae_first_date': mae_first,
                                    'mae_final_date': mae_final,
                                    'iou': iou_value
                                }
                            
                            # Use processed data from backend if available
                            if 'processed_data' in info and info['processed_data']:
                                # Combine all processed data into one DataFrame for display
                                all_data = []
                                for file_data in info['processed_data']:
                                    all_data.extend(file_data['data'])
                                
                                # Convert to DataFrame
                                df = pd.DataFrame(all_data)
                                st.session_state.data = df
                                st.session_state.file_processed = True
                                st.success(f"✅ Backend processed data loaded: {len(df):,} rows and {len(df.columns)} columns.")
                                st.rerun()
                            else:
                                st.session_state.file_processed = False
                        else:
                            st.session_state.file_processed = False
                    else:
                        # Backend error
                        st.error(f"Backend API error ({response.status_code}): {response.text}")
                        st.session_state.file_processed = False
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend API. Please ensure the backend server is running at http://0.0.0.0:3005")
                    st.session_state.file_processed = False
                    
                except Exception as e:
                    st.error(f"❌ Failed to send files to backend: {str(e)}")
                    st.session_state.file_processed = False
    
    elif not all_files_uploaded:
        # Show instructions when not all files are uploaded
        st.info("👆 Please upload all 3 Excel files in the specified order to get started")
        
        # Show which files are missing
        missing_files = [name for file, name in zip(files_uploaded, file_names) if not file]
        if missing_files:
            st.warning(f"Missing files: {', '.join(missing_files)}")
    
    # Display data if available
    if st.session_state.data is not None and st.session_state.file_processed:
        # Show processing metrics prominently if available
        if st.session_state.get('metrics') is not None:
            st.header("📊 Processing Results")
            
            # Create metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_rows = len(st.session_state.data)
                st.metric(
                    label="Total Rows Processed",
                    value=f"{total_rows:,}",
                    help="Total number of rows in the processed dataset"
                )
            
            with col2:
                metrics = st.session_state.metrics
                mae_first = metrics.get('mae_first_date', 'N/A') if metrics else 'N/A'
                st.metric(
                    label="MAE First Date",
                    value=mae_first if mae_first != 'N/A' else 'Not Available',
                    help="Mean Absolute Error for the first date"
                )
            
            with col3:
                mae_final = metrics.get('mae_final_date', 'N/A') if metrics else 'N/A'
                st.metric(
                    label="MAE Final Date",
                    value=mae_final if mae_final != 'N/A' else 'Not Available',
                    help="Mean Absolute Error for the final date"
                )
            
            with col4:
                iou_value = metrics.get('iou', 'N/A') if metrics else 'N/A'
                if isinstance(iou_value, (int, float)):
                    iou_display = f"{iou_value:.4f}"
                else:
                    iou_display = str(iou_value) if iou_value != 'N/A' else 'Not Available'
                
                st.metric(
                    label="IoU Score",
                    value=iou_display,
                    help="Intersection over Union score (higher is better)"
                )
            
            st.divider()
        
        st.header("📋 Data View")
        
        df = st.session_state.data
        
        # Display dataset information
        display_data_info(df, st.session_state.current_page, rows_per_page)
        
        st.divider()
        
        # Pagination controls (top)
        st.subheader("Navigation")
        new_page = display_pagination_controls(
            len(df), 
            rows_per_page, 
            st.session_state.current_page,
            "top"
        )
        
        # Update current page if changed
        if new_page != st.session_state.current_page:
            st.session_state.current_page = new_page
            st.rerun()
        
        # Calculate data slice for current page
        start_idx = st.session_state.current_page * rows_per_page
        end_idx = start_idx + rows_per_page
        current_data = df.iloc[start_idx:end_idx]
        
        # Display the data table with hover tooltips
        col_title, col_nav = st.columns([3, 1])
        
        with col_title:
            st.subheader(f"Page {st.session_state.current_page + 1}")
        
        with col_nav:
            # Quick navigation buttons
            total_pages = math.ceil(len(df) / rows_per_page)
            col_prev, col_next = st.columns(2)
            
            with col_prev:
                if st.button("⬅️ Prev", disabled=st.session_state.current_page == 0, key="quick_prev"):
                    st.session_state.current_page = max(0, st.session_state.current_page - 1)
                    st.rerun()
            
            with col_next:
                if st.button("Next ➡️", disabled=st.session_state.current_page >= total_pages - 1, key="quick_next"):
                    st.session_state.current_page = min(total_pages - 1, st.session_state.current_page + 1)
                    st.rerun()
        
        # Show row numbers for context
        display_df = current_data.copy()
        display_df = display_df.reset_index(drop=True)
        display_df.index = display_df.index + start_idx + 1
        
        # Display interactive data table with row selection to show popup
        if len(df.columns) > 0:
            last_column_name = df.columns[-2:]
            
            # Add a selection column to make rows clickable
            display_df_with_selection = display_df.copy()
            display_df_with_selection.insert(0, "👆 Click", False)
            
            # Use data_editor for interactive selection
            st.write("**Click on any row's checkbox to see detailed popup:**")
            
            edited_df = st.data_editor(
                display_df_with_selection,
                use_container_width=True,
                height=600,
                hide_index=False,
                column_config={
                    "👆 Click": st.column_config.CheckboxColumn(
                        "Select Row",
                        help="Click to view row details",
                        default=False,
                    )
                }
            )
            
            # Check which rows are selected and show popup
            selected_rows = edited_df[edited_df["👆 Click"]]
            
            if len(selected_rows) > 0:
                # Show popup for selected rows
                for idx, (_, row) in enumerate(selected_rows.iterrows()):
                    row_data = row.drop("👆 Click")  # Remove the selection column
                    row_number = row.name
                    
                    # Create popup-like container
                    with st.container():
                        st.success(f"📋 **Row {row_number} Details:**")
                        
                        # Create columns for better layout
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.write("**All Columns:**")
                            for col_name in row_data.index:
                                if col_name in last_column_name:  # Don't repeat last column
                                    value = row_data[col_name]
                                    display_value = str(value) if value is not None else "N/A"
                                    st.write(f"**{col_name}:** {display_value}")
                        
                        # with col2:
                        #     # Highlight the last column (main popup content)
                        #     st.markdown("### 🎯 Highlighted Information:")
                        #     last_col_value = row_data[last_column_name]
                        #     last_col_display = str(last_col_value) if last_col_value is not None else "N/A"
                        #     st.info(f"**{last_column_name}:** {last_col_display}")
                        
                        st.divider()
            else:
                # Show instruction when no rows selected
                st.info("💡 Click on the checkbox in any row to view its detailed information")
                
        else:
            # Fallback for tables with no columns
            st.dataframe(
                display_df,
                use_container_width=True,
                height=600
            )
        
        # Quick navigation after table
        st.markdown("---")
        col_info, col_quick_nav = st.columns([2, 1])
        
        with col_info:
            total_pages = math.ceil(len(df) / rows_per_page)
            st.write(f"**Page {st.session_state.current_page + 1} of {total_pages}** | Showing rows {start_idx + 1}-{min(end_idx, len(df))} of {len(df):,}")
        
        with col_quick_nav:
            # Large navigation buttons for easy access
            nav_col1, nav_col2, nav_col3 = st.columns(3)
            
            with nav_col1:
                if st.button("⏮️ First", disabled=st.session_state.current_page == 0, key="table_first"):
                    st.session_state.current_page = 0
                    st.rerun()
            
            with nav_col2:
                if st.button("⬅️ Previous", disabled=st.session_state.current_page == 0, key="table_prev"):
                    st.session_state.current_page = max(0, st.session_state.current_page - 1)
                    st.rerun()
            
            with nav_col3:
                if st.button("Next ➡️", disabled=st.session_state.current_page >= total_pages - 1, key="table_next", type="primary"):
                    st.session_state.current_page = min(total_pages - 1, st.session_state.current_page + 1)
                    st.rerun()
        # Pagination controls (bottom)
        st.divider()
        new_page_bottom = display_pagination_controls(
            len(df), 
            rows_per_page, 
            st.session_state.current_page,
            "bottom"
        )
        
        # Update current page if changed from bottom controls
        if new_page_bottom != st.session_state.current_page:
            st.session_state.current_page = new_page_bottom
            st.rerun()
        
        # Download Options
        st.header("📥 Download Options")
        
        # Create download buttons in a grid layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download current page as CSV
            csv_data = current_data.to_csv(index=False)
            st.download_button(
                label="📄 Download Current Page (CSV)",
                data=csv_data,
                file_name=f"data_page_{st.session_state.current_page + 1}.csv",
                mime="text/csv",
                type="primary",
                width="stretch"
            )
        
        with col2:
            # Download full dataset as CSV
            full_csv_data = df.to_csv(index=False)
            st.download_button(
                label="📊 Download Full Dataset (CSV)",
                data=full_csv_data,
                file_name="full_dataset.csv",
                mime="text/csv",
                type="primary", 
                width="stretch"
            )
            
        with col3:
            # Download full dataset as Excel
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, sheet_name='Data', engine='openpyxl')
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                label="📗 Download as Excel (.xlsx)",
                data=excel_data,
                file_name="dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                width="stretch"
            )
    
    else:
        # Show instructions when no data is available
        st.info("👆 Please upload all 3 Excel files and click 'PROCESS ALL FILES' to get started")
        
        # Show example of supported formats
        with st.expander("ℹ️ Supported File Formats"):
            st.write("""
            **Supported Excel formats:**
            - `.xlsx` (Excel 2007+)
            - `.xls` (Excel 97-2003)
            
            **Features:**
            - ✅ Automatic data type detection
            - ✅ Large file support with pagination
            - ✅ Memory-efficient processing
            - ✅ Export capabilities
            """)

if __name__ == "__main__":
    main()
