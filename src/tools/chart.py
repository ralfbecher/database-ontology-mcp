"""Chart generation tool."""

import logging
import base64
import io
from datetime import datetime
from typing import Dict, List, Any, Optional

import mcp.types as types
from ..chart_utils import create_plotly_chart, create_matplotlib_chart, save_image_to_tmp, create_error_response

logger = logging.getLogger(__name__)


def generate_chart(
    data_source: List[Dict[str, Any]],
    chart_type: str,
    x_column: str,
    y_column: Optional[str] = None,
    color_column: Optional[str] = None,
    title: Optional[str] = None,
    chart_library: str = "matplotlib",
    chart_style: str = "grouped",
    width: int = 800,
    height: int = 600
) -> list[types.ContentBlock]:
    """Generate chart implementation. Full documentation in main.py."""
    try:
        # Check for visualization libraries with detailed guidance
        missing_libs = []
        
        try:
            import pandas as pd
        except ImportError:
            missing_libs.append("pandas")
            
        if chart_library == "plotly":
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                from plotly.io import to_html, to_image
            except ImportError:
                missing_libs.append("plotly")
        else:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                plt.style.use('default')
            except ImportError as e:
                if "matplotlib" in str(e):
                    missing_libs.append("matplotlib")
                if "seaborn" in str(e):
                    missing_libs.append("seaborn")
        
        if missing_libs:
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ Missing required visualization libraries: {', '.join(missing_libs)}. Install with: pip install {' '.join(missing_libs)}"
                )
            ]
        
        # Validate input data
        if not data_source:
            return [
                types.TextContent(
                    type="text",
                    text="❌ No data provided for charting"
                )
            ]
        
        data = data_source
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(data)
        
        # Validate required columns
        if x_column not in df.columns:
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ X-axis column '{x_column}' not found in data. Available columns: {list(df.columns)}"
                )
            ]
        
        if chart_type in ["bar", "line", "scatter"] and y_column and y_column not in df.columns:
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ Y-axis column '{y_column}' not found in data. Available columns: {list(df.columns)}"
                )
            ]
        
        # Generate title if not provided
        if not title:
            if chart_type == "bar":
                title = f"{y_column or 'Count'} by {x_column}"
            elif chart_type == "line":
                title = f"{y_column} over {x_column}"
            elif chart_type == "scatter":
                title = f"{y_column} vs {x_column}"
            elif chart_type == "heatmap":
                title = f"Heatmap of {x_column}" + (f" and {y_column}" if y_column else "")
            else:
                title = f"Chart of {x_column}"
        
        # Create chart ID for file naming and logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c for c in title.replace(" ", "_") if c.isalnum() or c in "_-")
        chart_id = f"{chart_type}_{safe_title}_{timestamp}"
        
        # Generate chart and create PNG image for Claude Desktop
        image_data = None
        image_file_path = None
        
        if chart_library == "plotly":
            try:
                fig = create_plotly_chart(df, chart_type, x_column, y_column, color_column, title, chart_style, width, height)
                # Try to export as PNG
                try:
                    image_bytes = fig.to_image(format='png', width=width, height=height, scale=2)
                    image_data = base64.b64encode(image_bytes).decode('utf-8')
                    # Save PNG to tmp directory
                    image_file_path = save_image_to_tmp(image_bytes, chart_id, 'png')
                except Exception as e:
                    if "kaleido" in str(e).lower():
                        # Fallback to matplotlib if kaleido not available
                        chart_library = "matplotlib"
                    else:
                        raise e
            except ImportError:
                # Plotly not available, fall back to matplotlib
                chart_library = "matplotlib"
        
        if chart_library == "matplotlib":
            fig = create_matplotlib_chart(df, chart_type, x_column, y_column, color_column, title, chart_style, width, height)
            # Generate PNG bytes with optimized settings
            import io
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight', 
                       facecolor='white', edgecolor='none', transparent=False,
                       dpi=150, pad_inches=0.1)
            image_bytes = img_buffer.getvalue()
            img_buffer.close()
            
            # Convert to base64 for Claude Desktop
            image_data = base64.b64encode(image_bytes).decode('utf-8')
            
            # Save PNG to tmp directory
            image_file_path = save_image_to_tmp(image_bytes, chart_id, 'png')
            
            # Close figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)
        
        
        # Log chart creation success
        logger.info(f"Created {chart_type} chart with {len(df)} data points using {chart_library}")
        
        # Return the chart as PNG image for Claude Desktop
        if image_data:
            # Include file path info if image was saved
            file_info = f" (PNG saved to: {image_file_path})" if image_file_path else ""
            
            return [
                types.TextContent(
                    type="text", 
                    text=f"Generated {chart_type} chart with {len(df)} data points using {chart_library}.{file_info}"
                ),
                types.ImageContent(
                    type="image",
                    data=image_data,
                    mimeType="image/png"
                )
            ]
        else:
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ Failed to generate chart image"
                )
            ]
        
    except Exception as e:
        logger.error(f"Chart creation error: {e}")
        return [
            types.TextContent(
                type="text",
                text=f"❌ Chart generation failed: {str(e)}"
            )
        ]