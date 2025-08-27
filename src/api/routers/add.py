from __future__ import annotations

from typing import List

from fastapi import APIRouter
from fastapi import Form
from fastapi import status
from fastapi import UploadFile

from api.helpers.exception_handler import ExceptionHandler
from api.helpers.exception_handler import ResponseMessage
from shared.logging import get_logger

logger = get_logger(__name__)

add_router = APIRouter()


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
        
        for file in valid_files:
            # Here you would implement your Excel processing logic
            # For now, we'll simulate the processing
            
            # Read file content
            content = await file.read()
            
            # Simulate processing (you would replace this with actual Excel processing)
            # Example: using pandas to read Excel
            # import pandas as pd
            # import io
            # df = pd.read_excel(io.BytesIO(content))
            # total_rows += len(df)
            
            # For simulation, let's assume each file has 50 rows
            total_rows += 50
            processed_files.append(file.filename)
            
            # Reset file pointer for potential future use
            await file.seek(0)

        # Prepare success response
        output = {
            'processed_files': processed_files,
            'total_rows_processed': total_rows,
            'processing_status': 'completed',
        }

        return exception_handler.handle_success(output)

    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during Excel file processing: {str(e)}',
            extra={
                'files': [file.filename for file in valid_files]
            },
        )
