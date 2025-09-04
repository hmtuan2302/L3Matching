from __future__ import annotations

from typing import List
import pandas as pd
import numpy as np
import io
import datetime

from fastapi import APIRouter
from fastapi import Form
from fastapi import status
from fastapi import UploadFile

from api.helpers.exception_handler import ExceptionHandler
from api.helpers.exception_handler import ResponseMessage
from shared.logging import get_logger
from application import AddApplication, AddApplicationInput, AddApplicationOutput

logger = get_logger(__name__)

add_router = APIRouter()


def convert_datetime_to_string(obj):
    """Convert datetime objects to string representation."""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.strftime('%Y-%m-%d %H:%M:%S') if hasattr(obj, 'hour') else obj.strftime('%Y-%m-%d')
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, np.ndarray):
        # Convert numpy arrays to lists
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        # Convert numpy scalars to Python scalars
        return obj.item()
    elif isinstance(obj, np.bool_):
        # Convert numpy boolean to Python boolean
        return bool(obj)
    return obj


def clean_dataframe_for_json(df):
    """Clean DataFrame to make it JSON serializable."""
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Convert all datetime columns to string
    for col in df_clean.columns:
        if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_clean[col] = df_clean[col].replace('NaT', None)
        elif col.lower() == 'explanations':
            # Convert explanations column to string format
            df_clean[col] = df_clean[col].apply(lambda x: str(x) if x is not None else None)
        elif df_clean[col].dtype == 'object':
            # Check if any values are datetime objects or numpy arrays
            df_clean[col] = df_clean[col].apply(convert_datetime_to_string)
        elif str(df_clean[col].dtype).startswith('float') or str(df_clean[col].dtype).startswith('int'):
            # Convert numpy numeric types to Python types
            df_clean[col] = df_clean[col].apply(lambda x: x.item() if pd.notnull(x) and hasattr(x, 'item') else x)

    # Reset index to avoid datetime index as key
    df_clean = df_clean.reset_index(drop=True)
    
    # Ensure index is not datetime
    if pd.api.types.is_datetime64_any_dtype(df_clean.index):
        df_clean.index = df_clean.index.strftime('%Y-%m-%d %H:%M:%S')

    # Replace NaN values with None for JSON serialization
    df_clean = df_clean.astype(object).where(pd.notnull(df_clean), None)
    
    return df_clean


def validate_excel_file(file: UploadFile) -> bool:
    """Validate if the uploaded file is an Excel file."""
    allowed_extensions = ['.xlsx', '.xls']
    allowed_mime_types = [
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel',
    ]
    
    # Check file extension
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return False
    
    # Check MIME type
    if file.content_type not in allowed_mime_types:
        return False
    
    return True


@add_router.post(
    '/upload-excel',
    response_model=None,
    tags=['Files'],
    responses={
        status.HTTP_200_OK: {
            'description': ResponseMessage.SUCCESS,
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.SUCCESS,
                        'info': {
                            'processed_files': [
                                'example_1.xlsx',
                                'example_2.xls',
                            ],
                            'total_rows_processed': 150,
                            'processing_status': 'completed',
                        },
                    },
                },
            },
        },
        status.HTTP_400_BAD_REQUEST: {
            'description': ResponseMessage.BAD_REQUEST,
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.BAD_REQUEST,
                    },
                },
            },
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            'description': ResponseMessage.UNPROCESSABLE_ENTITY,
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.UNPROCESSABLE_ENTITY,
                    },
                },
            },
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            'description': ResponseMessage.INTERNAL_SERVER_ERROR,
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.INTERNAL_SERVER_ERROR,
                    },
                },
            },
        },
    },
    description="""
    Upload and process Excel files (.xlsx, .xls).\n
    Input parameters:\n
    - file: Excel file to upload
    Output parameters:\n
    - message: Success or error message
    - info: Processing results including:
      + processed_files: List of successfully processed file names
      + total_rows_processed: Total number of rows processed
      + processing_status: Status of the processing operation
    """,
)
async def upload_excel_files(
    file: UploadFile
):
    """Upload and process Excel files."""
    
    exception_handler = ExceptionHandler(
        logger=logger.bind(),  # We'll need to create a proper logger
        service_name=__name__,
    )

    # Validate input
    if not file:
        return exception_handler.handle_bad_request(
            message='At least one file must be provided',
        )

    # Validate Excel files
    invalid_files = []
    valid_files = []
    
    if not validate_excel_file(file):
        invalid_files.append(file.filename)
    else:
        valid_files.append(file)

    if invalid_files:
        return exception_handler.handle_unprocessable_entity(
            message=f'Invalid Excel files detected: {", ".join(invalid_files)}',
            extra={'invalid_files': invalid_files},
        )

    try:
        # Process Excel files
        processed_files = []
        total_rows = 0
        processed_data = []
        
        for file in valid_files:
            # Read file content
            content = await file.read()
            
            # Use pandas to read Excel file
            df = pd.read_excel(io.BytesIO(content), header=None)
            
            # Add new "test" column
            df['test'] = f'processed_{file.filename}'
            
            # Convert all datetime columns to string
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    df[col] = df[col].replace('NaT', None)

            # Reset index to avoid datetime index as key
            df = df.reset_index(drop=True)
    
            # Replace NaN values with None for JSON serialization
            # df = df.where(pd.notnull(df), None)
            df = df.astype(object).where(pd.notnull(df), None)
            
            # Count rows
            total_rows += len(df)
            processed_files.append(file.filename)
            
            # Convert DataFrame to dict for JSON response
            processed_data.append({
                'filename': file.filename,
                'data': df.to_dict('records'),  # Convert to list of dictionaries
                'columns': df.columns.tolist(),
                'shape': df.shape
            })
            
            # Reset file pointer for potential future use
            await file.seek(0)

        # Prepare success response
        output = {
            'processed_files': processed_files,
            'total_rows_processed': total_rows,
            'processing_status': 'completed',
            'processed_data': processed_data  # Include the processed data
        }

        return exception_handler.handle_success(output)

    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during Excel file processing: {str(e)}',
            extra={
                'files': [file.filename for file in valid_files]
            },
        )


@add_router.post(
    '/embedding',
    response_model=None,
    tags=['Files'],
    responses={
        status.HTTP_200_OK: {
            'description': ResponseMessage.SUCCESS,
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.SUCCESS,
                        'info': {
                            'processing_status': 'completed',
                        },
                    },
                },
            },
        },
        status.HTTP_400_BAD_REQUEST: {
            'description': ResponseMessage.BAD_REQUEST,
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.BAD_REQUEST,
                    },
                },
            },
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            'description': ResponseMessage.UNPROCESSABLE_ENTITY,
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.UNPROCESSABLE_ENTITY,
                    },
                },
            },
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            'description': ResponseMessage.INTERNAL_SERVER_ERROR,
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.INTERNAL_SERVER_ERROR,
                    },
                },
            },
        },
    },
    description="""
    Embedding
    """,
)
async def upload_excel_files(
    file: UploadFile
):
    """Upload and process Excel files."""
    
    exception_handler = ExceptionHandler(
        logger=logger.bind(),  # We'll need to create a proper logger
        service_name=__name__,
    )

    # Validate input
    if not file:
        return exception_handler.handle_bad_request(
            message='At least one file must be provided',
        )

    # Validate Excel files
    invalid_files = []
    valid_files = []
    
    if not validate_excel_file(file):
        invalid_files.append(file.filename)
    else:
        valid_files.append(file)

    if invalid_files:
        return exception_handler.handle_unprocessable_entity(
            message=f'Invalid Excel files detected: {", ".join(invalid_files)}',
            extra={'invalid_files': invalid_files},
        )

    try:
        # Process Excel files
        processed_files = []
        total_rows = 0
        processed_data = []
        
        for file in valid_files:
            # Read file content
            content = await file.read()
            
            # Use pandas to read Excel file
            df = pd.read_excel(io.BytesIO(content), header=None)
            
            # Add new "test" column
            df['test'] = f'processed_{file.filename}'
            
            # Convert all datetime columns to string
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    df[col] = df[col].replace('NaT', None)

            # Reset index to avoid datetime index as key
            df = df.reset_index(drop=True)
    
            # Replace NaN values with None for JSON serialization
            # df = df.where(pd.notnull(df), None)
            df = df.astype(object).where(pd.notnull(df), None)
            
            # Count rows
            total_rows += len(df)
            processed_files.append(file.filename)
            
            # Convert DataFrame to dict for JSON response
            processed_data.append({
                'filename': file.filename,
                'data': df.to_dict('records'),  # Convert to list of dictionaries
                'columns': df.columns.tolist(),
                'shape': df.shape
            })
            
            # Reset file pointer for potential future use
            await file.seek(0)

        # Prepare success response
        output = {
            'processed_files': processed_files,
            'total_rows_processed': total_rows,
            'processing_status': 'completed',
            'processed_data': processed_data  # Include the processed data
        }

        return exception_handler.handle_success(output)

    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during Excel file processing: {str(e)}',
            extra={
                'files': [file.filename for file in valid_files]
            },
        )


@add_router.post(
    '/upload-multiple-excel',
    response_model=None,
    tags=['Files'],
    responses={
        status.HTTP_200_OK: {
            'description': ResponseMessage.SUCCESS,
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.SUCCESS,
                        'info': {
                            'processed_files': [
                                'mdl_history.xlsx',
                                'mdl_input.xlsx',
                                'l3_file.xlsx',
                            ],
                            'total_rows_processed': 450,
                            'processing_status': 'completed',
                            'file_order': ['mdl_history', 'mdl_input', 'l3']
                        },
                    },
                },
            },
        },
        status.HTTP_400_BAD_REQUEST: {
            'description': ResponseMessage.BAD_REQUEST,
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.BAD_REQUEST,
                    },
                },
            },
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            'description': ResponseMessage.UNPROCESSABLE_ENTITY,
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.UNPROCESSABLE_ENTITY,
                    },
                },
            },
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            'description': ResponseMessage.INTERNAL_SERVER_ERROR,
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.INTERNAL_SERVER_ERROR,
                    },
                },
            },
        },
    },
    description="""
    Upload and process 3 Excel files in a specific order: MDL History, MDL Input, L3.\n
    Input parameters:\n
    - mdl_history: MDL History Excel file
    - mdl_input: MDL Input Excel file  
    - l3_file: L3 Excel file
    Output parameters:\n
    - message: Success or error message
    - info: Processing results including:
      + processed_files: List of successfully processed file names in order
      + total_rows_processed: Total number of rows processed across all files
      + processing_status: Status of the processing operation
      + file_order: Order in which files were processed
    """,
)
async def upload_multiple_excel_files(
    mdl_history: UploadFile,
    mdl_input: UploadFile,
    l3_file: UploadFile
):
    """Upload and process 3 Excel files in order: MDL History, MDL Input, L3."""
    
    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )

    # Validate input - all files must be provided
    files = [mdl_history, mdl_input, l3_file]
    file_names = ['mdl_history', 'mdl_input', 'l3_file']
    
    if not all(files):
        return exception_handler.handle_bad_request(
            message='All three files must be provided: mdl_history, mdl_input, l3_file',
            extra={'missing_files': [name for name, file in zip(file_names, files) if not file]}
        )

    # Validate Excel files
    invalid_files = []
    valid_files = []
    
    for i, file in enumerate(files):
        if not validate_excel_file(file):
            invalid_files.append(f"{file_names[i]}: {file.filename}")
        else:
            valid_files.append((file_names[i], file))

    if invalid_files:
        return exception_handler.handle_unprocessable_entity(
            message=f'Invalid Excel files detected: {", ".join(invalid_files)}',
            extra={'invalid_files': invalid_files},
        )

    try:
        input = AddApplicationInput(
            files=list([mdl_history, mdl_input, l3_file])
        )
        result = AddApplication().run(input)
        
        # Convert Polars DataFrame result to format suitable for Streamlit
        processed_files = [file.filename for file in files]
        file_order = ['mdl_history', 'mdl_input', 'l3_file']
        
        # Check if result has the expected output structure
        if hasattr(result, 'result') and result.result is not None:
            # Convert Polars DataFrame to pandas for JSON serialization
            df_pandas = result.result.to_pandas()
            
            # Clean DataFrame for JSON serialization
            df_clean = clean_dataframe_for_json(df_pandas)
            
            # Prepare processed data for response
            processed_data = [{
                'file_type': 'processed_output',
                'filename': 'processed_result.xlsx',
                'data': df_clean.to_dict('records'),
                'columns': df_clean.columns.tolist(),
                'shape': df_clean.shape,
                'processing_order': 1
            }]
            
            total_rows_processed = len(df_clean)
            
        else:
            # Fallback: return empty data if no output DataFrame
            processed_data = []
            total_rows_processed = 0

        # Prepare success response
        output = {
            'processed_files': processed_files,
            'total_rows_processed': total_rows_processed,
            'processing_status': 'completed',
            'file_order': file_order,
            'processed_data': processed_data,
            'mae_first_date': result.mae_first_date,
            'mae_final_date': result.mae_final_date,
            'iou': result.iou
        }
        
        return exception_handler.handle_success(output)

    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during multiple Excel files processing: {str(e)}',
            extra={
                'files': [f"{file_type}: {file.filename}" for file_type, file in valid_files]
            },
        )