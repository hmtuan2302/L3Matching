# Excel Data Viewer

## Overview

A Streamlit-based web application for viewing and browsing Excel files with pagination controls. The application allows users to upload Excel files (.xlsx or .xls) and view their contents in a paginated, user-friendly interface. This is a single-page application focused on data visualization and file handling.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit - chosen for rapid prototyping and built-in web UI components
- **Layout**: Wide layout configuration for better data table display
- **Components**: File uploader, data display tables, pagination controls
- **Styling**: Uses Streamlit's default theming with custom page configuration

### Data Processing Layer
- **File Handling**: Pandas for Excel file reading and DataFrame operations
- **Supported Formats**: Excel files (.xlsx, .xls) with automatic format detection
- **Error Handling**: Comprehensive exception handling for file upload and processing errors
- **Memory Management**: Stream-based file processing using io module

### User Interface Design
- **Pagination System**: Custom pagination controls to handle large datasets efficiently
- **Responsive Display**: Wide page layout optimized for data table viewing
- **File Upload**: Streamlit's native file uploader component
- **Error Messaging**: User-friendly error messages for file format and processing issues

### Code Organization
- **Modular Functions**: Separate functions for file loading, pagination, and display logic
- **Type Hints**: Python type annotations for better code maintainability
- **Single File Architecture**: All functionality contained in app.py for simplicity

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the user interface
- **Pandas**: Data manipulation and Excel file reading capabilities
- **io**: Built-in Python module for stream handling

### File Format Support
- **Excel Files**: Native support for .xlsx and .xls formats through pandas
- **No Database**: Application operates entirely in memory without persistent storage

### Runtime Environment
- **Python**: Core runtime environment
- **No External APIs**: Self-contained application with no third-party service integrations
- **No Authentication**: Open access application without user management