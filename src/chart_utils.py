"""Chart generation utilities for the Database Ontology MCP Server."""

import logging
import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Temporary directory for file storage
TMP_DIR = Path(__file__).parent.parent / "tmp"


def create_plotly_chart(df, chart_type, x_column, y_column, color_column, title, chart_style, width=800, height=600):
    """Create Plotly chart based on type.

    Args:
        y_column: Can be a string (single measure) or list of strings (multiple measures for line charts)
    """
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    if chart_type == "bar":
        if chart_style == "stacked" and color_column:
            # Stacked bar chart with two dimensions: x_column (categories) and color_column (stack groups)
            # Aggregate data first to handle duplicates
            agg_df = df.groupby([x_column, color_column], as_index=False)[y_column].sum()
            # Sort by sum of stacked values (descending)
            total_by_category = agg_df.groupby(x_column)[y_column].sum().sort_values(ascending=False)
            category_order = total_by_category.index.tolist()
            fig = px.bar(agg_df, x=x_column, y=y_column, color=color_column, title=title,
                        barmode='stack', category_orders={x_column: category_order})
        elif color_column:
            # Grouped bar chart - aggregate data first to handle duplicates
            agg_df = df.groupby([x_column, color_column], as_index=False)[y_column].sum()
            # Sort by sum of grouped values (descending)
            total_by_category = agg_df.groupby(x_column)[y_column].sum().sort_values(ascending=False)
            category_order = total_by_category.index.tolist()
            fig = px.bar(agg_df, x=x_column, y=y_column, color=color_column, title=title,
                        barmode='group', category_orders={x_column: category_order})
            # Make bars wider for grouped charts
            fig.update_layout(bargap=0.15, bargroupgap=0.05)
        else:
            # Simple bar chart - sort by measure value (descending)
            sorted_df = df.sort_values(by=y_column, ascending=False)
            fig = px.bar(sorted_df, x=x_column, y=y_column, title=title)
    elif chart_type == "line":
        # Support multiple measures for line charts
        if isinstance(y_column, list):
            # Multiple measures - create separate line for each measure
            fig = go.Figure()
            for measure in y_column:
                if measure not in df.columns:
                    continue
                fig.add_trace(go.Scatter(
                    x=df[x_column],
                    y=df[measure],
                    mode='lines+markers',
                    name=measure
                ))
            fig.update_layout(
                title=title,
                xaxis_title=x_column,
                yaxis_title="Value",
                showlegend=True
            )
        else:
            # Single measure with optional color grouping
            fig = px.line(df, x=x_column, y=y_column, color=color_column, title=title)

        # Enhance for time series
        if df[x_column].dtype in ['datetime64[ns]', 'object']:
            try:
                df[x_column] = pd.to_datetime(df[x_column])
                fig.update_xaxes(title=x_column, type='date')
            except:
                pass
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title=title,
                        size_max=15)
    elif chart_type == "heatmap":
        if y_column:
            # Pivot table heatmap
            pivot_df = df.pivot_table(index=x_column, columns=y_column, aggfunc='size', fill_value=0)
        else:
            # Correlation heatmap
            numeric_cols = df.select_dtypes(include=['number']).columns
            pivot_df = df[numeric_cols].corr()
        
        fig = px.imshow(pivot_df, title=title, aspect="auto")
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    
    # Check if x-axis labels are long and need rotation
    if chart_type in ["bar", "line", "scatter"]:
        x_labels = df[x_column].astype(str).unique()
        max_label_length = max([len(str(label)) for label in x_labels]) if len(x_labels) > 0 else 0
        
        if max_label_length > 10 or len(x_labels) > 8:
            # Rotate x-axis labels for better readability
            fig.update_xaxes(tickangle=45)
    
    # Apply consistent styling
    # Determine if we should show legend
    show_legend = color_column or (chart_type == "line" and isinstance(y_column, list))

    fig.update_layout(
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(b=100, t=60, l=60, r=200),  # Extra right margin for legend
        showlegend=show_legend,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02  # Position legend just outside the plot area on the right
        ),
        width=width,
        height=height
    )
    
    return fig


def create_matplotlib_chart(df, chart_type, x_column, y_column, color_column, title, chart_style, width, height):
    """Create Matplotlib/Seaborn chart based on type.

    Args:
        y_column: Can be a string (single measure) or list of strings (multiple measures for line charts)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set figure size
    fig, ax = plt.subplots(figsize=(width/100, height/100))

    if chart_type == "bar":
        if color_column and chart_style == "stacked":
            # Stacked bar chart with two dimensions: x_column and color_column
            # Sort by sum of stacked values (descending)
            pivot_df = df.pivot_table(index=x_column, columns=color_column, values=y_column, aggfunc='sum', fill_value=0)
            total_by_category = pivot_df.sum(axis=1).sort_values(ascending=False)
            pivot_df = pivot_df.loc[total_by_category.index]
            pivot_df.plot(kind='bar', stacked=True, ax=ax)
        elif color_column:
            # Grouped bar chart - use pivot table to handle multiple rows per x/hue combination
            # This aggregates (sums) values when there are duplicates
            pivot_df = df.pivot_table(index=x_column, columns=color_column, values=y_column, aggfunc='sum', fill_value=0)
            # Sort by total across all groups (descending)
            total_by_category = pivot_df.sum(axis=1).sort_values(ascending=False)
            pivot_df = pivot_df.loc[total_by_category.index]
            # Make bars 2-3x wider for grouped charts (width closer to 1.0 = wider)
            pivot_df.plot(kind='bar', ax=ax, width=0.85)
        else:
            # Simple bar chart - sort by measure value (descending)
            sorted_df = df.sort_values(by=y_column, ascending=False)
            category_order = sorted_df[x_column].tolist()
            sns.barplot(data=sorted_df, x=x_column, y=y_column, ax=ax, order=category_order, estimator='sum', errorbar=None)
    elif chart_type == "line":
        # Support multiple measures for line charts
        if isinstance(y_column, list):
            # Multiple measures - create separate line for each measure
            for measure in y_column:
                if measure not in df.columns:
                    continue
                ax.plot(df[x_column], df[measure], label=measure, marker='o')
            ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
            ax.set_ylabel("Value")
        elif color_column:
            # Single measure with color grouping
            for group in df[color_column].unique():
                group_data = df[df[color_column] == group]
                ax.plot(group_data[x_column], group_data[y_column], label=group, marker='o')
            ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
        else:
            # Simple line chart
            ax.plot(df[x_column], df[y_column], marker='o')
    elif chart_type == "scatter":
        sns.scatterplot(data=df, x=x_column, y=y_column, hue=color_column, ax=ax, s=60)
    elif chart_type == "heatmap":
        if y_column:
            pivot_df = df.pivot_table(index=x_column, columns=y_column, aggfunc='size', fill_value=0)
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns
            pivot_df = df[numeric_cols].corr()
        sns.heatmap(pivot_df, annot=True, cmap='viridis', ax=ax)
    
    # Check if x-axis labels are long and need rotation
    if chart_type in ["bar", "line", "scatter"]:
        x_labels = [str(label) for label in ax.get_xticklabels()]
        if x_labels:  # Only if we have labels
            max_label_length = max([len(label.get_text()) for label in ax.get_xticklabels()]) if ax.get_xticklabels() else 0
            num_labels = len(ax.get_xticklabels())
            
            if max_label_length > 10 or num_labels > 8:
                # Rotate x-axis labels for better readability
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(x_column)
    if y_column:
        ax.set_ylabel(y_column)

    # Position any legend outside the plot area on the right
    legend = ax.get_legend()
    if legend is not None:
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)

    plt.tight_layout()
    return fig


def save_image_to_tmp(image_bytes: bytes, chart_id: str, format: str) -> Optional[str]:
    """Save image bytes to temporary directory and return file path."""
    try:
        image_filename = f"{chart_id}.{format}"
        image_file_path = TMP_DIR / image_filename
        
        with open(image_file_path, 'wb') as f:
            f.write(image_bytes)
        
        logger.debug(f"Saved {format.upper()} chart to: {image_file_path}")
        return str(image_file_path)
    except Exception as e:
        logger.warning(f"Failed to save {format.upper()} file: {e}")
        return None