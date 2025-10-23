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


def format_measure_name(measure: str) -> str:
    """Format measure name by removing underscores and applying title case.

    Args:
        measure: Raw measure name (e.g., "total_revenue", "profit_margin_pct")

    Returns:
        Formatted measure name (e.g., "Total Revenue", "Profit Margin Pct")
    """
    return measure.replace('_', ' ').title()


def create_plotly_chart(df, chart_type, x_column, y_column, color_column, title, chart_style, width=800, height=600, sort_by=None, sort_order=None):
    """Create Plotly chart based on type.

    Args:
        y_column: Can be a string (single measure) or list of strings (multiple measures for line charts)
        sort_by: Column to sort by. If None, uses automatic sorting based on chart type:
            - Bar/grouped/stacked: sorts by measure (y_column) descending
            - Line: sorts by dimension (x_column) ascending
            - Heatmap: sorts x_column ascending, y_column descending
        sort_order: 'ascending' or 'descending'. If None, uses automatic order based on chart type.
    """
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    if chart_type == "bar":
        # Determine sorting for bar charts
        # Default: sort by measure (y_column) descending
        effective_sort_by = sort_by if sort_by else y_column
        effective_sort_order = sort_order if sort_order else 'descending'
        sort_ascending = (effective_sort_order == 'ascending')

        if chart_style == "stacked" and color_column:
            # Stacked bar chart with two dimensions: x_column (categories) and color_column (stack groups)
            # Aggregate data first to handle duplicates
            agg_df = df.groupby([x_column, color_column], as_index=False)[y_column].sum()
            # Sort by aggregated measure (default) or specified column
            if effective_sort_by == y_column:
                total_by_category = agg_df.groupby(x_column)[y_column].sum().sort_values(ascending=sort_ascending)
            else:
                # Sort by x_column or another column
                total_by_category = agg_df.groupby(x_column)[effective_sort_by].sum().sort_values(ascending=sort_ascending)
            category_order = total_by_category.index.tolist()
            fig = px.bar(agg_df, x=x_column, y=y_column, color=color_column, title=title,
                        barmode='stack', category_orders={x_column: category_order})
        elif color_column:
            # Grouped bar chart - aggregate data first to handle duplicates
            agg_df = df.groupby([x_column, color_column], as_index=False)[y_column].sum()
            # Sort by aggregated measure (default) or specified column
            if effective_sort_by == y_column:
                total_by_category = agg_df.groupby(x_column)[y_column].sum().sort_values(ascending=sort_ascending)
            else:
                # Sort by x_column or another column
                total_by_category = agg_df.groupby(x_column)[effective_sort_by].sum().sort_values(ascending=sort_ascending)
            category_order = total_by_category.index.tolist()
            fig = px.bar(agg_df, x=x_column, y=y_column, color=color_column, title=title,
                        barmode='group', category_orders={x_column: category_order})
            # Make bars wider for grouped charts
            fig.update_layout(bargap=0.15, bargroupgap=0.05)
        else:
            # Simple bar chart - aggregate first, then sort by measure (default) or specified column
            agg_df = df.groupby(x_column, as_index=False)[y_column].sum()
            # Sort by aggregated measure (default) or specified column
            sorted_df = agg_df.sort_values(by=effective_sort_by, ascending=sort_ascending)
            category_order = sorted_df[x_column].tolist()
            fig = px.bar(sorted_df, x=x_column, y=y_column, title=title,
                        category_orders={x_column: category_order})
    elif chart_type == "line":
        # Work with a copy to avoid modifying the original dataframe
        sorted_df = df.copy()

        # Determine sorting for line charts
        # Default: sort by dimension (x_column) ascending
        effective_sort_by = sort_by if sort_by else x_column
        effective_sort_order = sort_order if sort_order else 'ascending'
        sort_ascending = (effective_sort_order == 'ascending')

        # Try to convert x_column to datetime if it looks like a date
        # This ensures proper chronological sorting
        try:
            sorted_df[x_column] = pd.to_datetime(sorted_df[x_column])
        except (ValueError, TypeError):
            # Not a datetime column, use as-is
            pass

        # Sort data by x-axis column (default) or specified column for proper line chart ordering
        sorted_df = sorted_df.sort_values(by=effective_sort_by, ascending=sort_ascending)

        # Support multiple measures for line charts
        if isinstance(y_column, list):
            # Multiple measures - separate into regular and percentage measures
            # Strip spaces and check case-insensitively for percentage measures
            pct_measures = [m for m in y_column if m in sorted_df.columns and (
                m.strip().lower().endswith('_pct') or
                m.strip().lower().endswith('_percent') or
                m.strip().lower().endswith('_percentage')
            )]
            value_measures = [m for m in y_column if m in sorted_df.columns and m not in pct_measures]

            fig = go.Figure()

            # Plot value measures on primary y-axis
            for measure in value_measures:
                fig.add_trace(go.Scatter(
                    x=sorted_df[x_column],
                    y=sorted_df[measure],
                    mode='lines+markers',
                    name=format_measure_name(measure),
                    yaxis='y',
                    showlegend=True
                ))

            # Plot percentage measures on secondary y-axis if we have both types
            for measure in pct_measures:
                fig.add_trace(go.Scatter(
                    x=sorted_df[x_column],
                    y=sorted_df[measure],
                    mode='lines+markers',
                    name=format_measure_name(measure),
                    yaxis='y2' if value_measures else 'y',
                    line=dict(dash='dash') if value_measures else None,
                    showlegend=True
                ))

            # Configure layout with dual y-axes if needed
            layout_config = {
                'title': title,
                'xaxis_title': x_column,
                'showlegend': True
            }

            if pct_measures and value_measures:
                # Dual y-axis configuration
                layout_config['yaxis'] = dict(
                    title=", ".join([format_measure_name(m) for m in value_measures]),
                    side='left'
                )
                layout_config['yaxis2'] = dict(
                    title=", ".join([format_measure_name(m) for m in pct_measures]),
                    side='right',
                    overlaying='y',
                    titlefont=dict(color='gray'),
                    tickfont=dict(color='gray'),
                    ticksuffix='%'
                )
            elif pct_measures:
                layout_config['yaxis_title'] = ", ".join([format_measure_name(m) for m in pct_measures])
            else:
                layout_config['yaxis_title'] = ", ".join([format_measure_name(m) for m in value_measures])

            fig.update_layout(**layout_config)
        else:
            # Single measure with optional color grouping
            fig = px.line(sorted_df, x=x_column, y=y_column, color=color_column, title=title)

        # Enhance for time series
        if sorted_df[x_column].dtype in ['datetime64[ns]']:
            fig.update_xaxes(title=x_column, type='date')
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title=title,
                        size_max=15)
    elif chart_type == "heatmap":
        if y_column:
            # Pivot table heatmap
            pivot_df = df.pivot_table(index=x_column, columns=y_column, aggfunc='size', fill_value=0)

            # Sort heatmap: x_column ascending (index), y_column descending (columns)
            # Default behavior, can be overridden
            if not sort_by or sort_by == x_column:
                # Sort index (x_column) - default ascending
                x_sort_order = sort_order if sort_order else 'ascending'
                pivot_df = pivot_df.sort_index(ascending=(x_sort_order == 'ascending'))

            # Sort columns (y_column) descending by default
            if y_column:
                # For columns, we always want descending unless explicitly overridden
                pivot_df = pivot_df.sort_index(axis=1, ascending=False)
        else:
            # Correlation heatmap
            numeric_cols = df.select_dtypes(include=['number']).columns
            pivot_df = df[numeric_cols].corr()

            # Sort correlation heatmap indices
            if sort_order:
                pivot_df = pivot_df.sort_index(ascending=(sort_order == 'ascending'))
                pivot_df = pivot_df.sort_index(axis=1, ascending=(sort_order == 'ascending'))

        fig = px.imshow(pivot_df, title=title, aspect="auto")
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    
    # Check if x-axis labels are long and need rotation
    if chart_type in ["bar", "line", "scatter"]:
        # Use the appropriate dataframe based on chart type
        check_df = sorted_df if chart_type == "line" else df
        x_labels = check_df[x_column].astype(str).unique()
        max_label_length = max([len(str(label)) for label in x_labels]) if len(x_labels) > 0 else 0

        if max_label_length > 10 or len(x_labels) > 8:
            # Rotate x-axis labels for better readability
            fig.update_xaxes(tickangle=45)
    
    # Apply consistent styling
    # Determine if we should show legend
    show_legend = bool(color_column or (chart_type == "line" and isinstance(y_column, list)))

    # For line charts with multiple measures, use horizontal legend between title and plot
    if chart_type == "line" and isinstance(y_column, list):
        legend_config = dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,  # Position at top of plot area (below title)
            xanchor="center",
            x=0.5
        )
        margin_config = dict(b=100, t=120, l=60, r=60)  # Extra top margin for title and legend space
    else:
        legend_config = dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02
        )
        margin_config = dict(b=100, t=60, l=60, r=200)  # Extra right margin for legend

    fig.update_layout(
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=margin_config,
        showlegend=show_legend,
        legend=legend_config,
        width=width,
        height=height
    )
    
    return fig


def create_matplotlib_chart(df, chart_type, x_column, y_column, color_column, title, chart_style, width, height, sort_by=None, sort_order=None):
    """Create Matplotlib/Seaborn chart based on type.

    Args:
        y_column: Can be a string (single measure) or list of strings (multiple measures for line charts)
        sort_by: Column to sort by. If None, uses automatic sorting based on chart type:
            - Bar/grouped/stacked: sorts by measure (y_column) descending
            - Line: sorts by dimension (x_column) ascending
            - Heatmap: sorts x_column ascending, y_column descending
        sort_order: 'ascending' or 'descending'. If None, uses automatic order based on chart type.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set figure size
    fig, ax = plt.subplots(figsize=(width/100, height/100))

    if chart_type == "bar":
        # Determine sorting for bar charts
        # Default: sort by measure (y_column) descending
        effective_sort_by = sort_by if sort_by else y_column
        effective_sort_order = sort_order if sort_order else 'descending'
        sort_ascending = (effective_sort_order == 'ascending')

        if color_column and chart_style == "stacked":
            # Stacked bar chart with two dimensions: x_column and color_column
            # Sort by aggregated measure (default) or specified column
            pivot_df = df.pivot_table(index=x_column, columns=color_column, values=y_column, aggfunc='sum', fill_value=0)
            if effective_sort_by == y_column:
                total_by_category = pivot_df.sum(axis=1).sort_values(ascending=sort_ascending)
            else:
                # If sorting by x_column, sort the index directly
                total_by_category = pivot_df.sum(axis=1)
                total_by_category = total_by_category.sort_index(ascending=sort_ascending)
            pivot_df = pivot_df.loc[total_by_category.index]
            pivot_df.plot(kind='bar', stacked=True, ax=ax)
        elif color_column:
            # Grouped bar chart - use pivot table to handle multiple rows per x/hue combination
            # This aggregates (sums) values when there are duplicates
            pivot_df = df.pivot_table(index=x_column, columns=color_column, values=y_column, aggfunc='sum', fill_value=0)
            # Sort by aggregated measure (default) or specified column
            if effective_sort_by == y_column:
                total_by_category = pivot_df.sum(axis=1).sort_values(ascending=sort_ascending)
            else:
                # If sorting by x_column, sort the index directly
                total_by_category = pivot_df.sum(axis=1)
                total_by_category = total_by_category.sort_index(ascending=sort_ascending)
            pivot_df = pivot_df.loc[total_by_category.index]
            # Make bars 2-3x wider for grouped charts (width closer to 1.0 = wider)
            pivot_df.plot(kind='bar', ax=ax, width=0.85)
        else:
            # Simple bar chart - aggregate first, then sort by measure (default) or specified column
            agg_df = df.groupby(x_column, as_index=False)[y_column].sum()
            # Sort by aggregated measure (default) or specified column
            sorted_df = agg_df.sort_values(by=effective_sort_by, ascending=sort_ascending)
            category_order = sorted_df[x_column].tolist()
            sns.barplot(data=sorted_df, x=x_column, y=y_column, ax=ax, order=category_order, estimator='sum', errorbar=None)
    elif chart_type == "line":
        # Work with a copy to avoid modifying the original dataframe
        import pandas as pd
        sorted_df = df.copy()

        # Determine sorting for line charts
        # Default: sort by dimension (x_column) ascending
        effective_sort_by = sort_by if sort_by else x_column
        effective_sort_order = sort_order if sort_order else 'ascending'
        sort_ascending = (effective_sort_order == 'ascending')

        # Try to convert x_column to datetime if it looks like a date
        # This ensures proper chronological sorting
        try:
            sorted_df[x_column] = pd.to_datetime(sorted_df[x_column])
        except (ValueError, TypeError):
            # Not a datetime column, use as-is
            pass

        # Sort data by x-axis column (default) or specified column for proper line chart ordering
        sorted_df = sorted_df.sort_values(by=effective_sort_by, ascending=sort_ascending)

        # Support multiple measures for line charts
        if isinstance(y_column, list):
            # Multiple measures - separate into regular and percentage measures
            # Strip spaces and check case-insensitively for percentage measures
            pct_measures = [m for m in y_column if m in sorted_df.columns and (
                m.strip().lower().endswith('_pct') or
                m.strip().lower().endswith('_percent') or
                m.strip().lower().endswith('_percentage')
            )]
            value_measures = [m for m in y_column if m in sorted_df.columns and m not in pct_measures]

            # Plot value measures on primary y-axis
            for measure in value_measures:
                ax.plot(sorted_df[x_column], sorted_df[measure], label=format_measure_name(measure), marker='o')

            # If we have both value and percentage measures, create secondary y-axis for percentages
            if pct_measures and value_measures:
                ax2 = ax.twinx()
                for measure in pct_measures:
                    ax2.plot(sorted_df[x_column], sorted_df[measure], label=format_measure_name(measure), marker='o', linestyle='--')
                ax2.set_ylabel(", ".join([format_measure_name(m) for m in pct_measures]), color='gray')
                ax2.tick_params(axis='y', labelcolor='gray')
                # Format y-axis ticks as percentages
                from matplotlib.ticker import FuncFormatter
                ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))

                # Combine legends from both axes - position between title and plot
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(0.5, 1.0), loc='lower center', ncol=3, borderaxespad=0.5)
                ax.set_ylabel(", ".join([format_measure_name(m) for m in value_measures]))
            elif pct_measures:
                # Only percentage measures
                for measure in pct_measures:
                    ax.plot(sorted_df[x_column], sorted_df[measure], label=format_measure_name(measure), marker='o')
                ax.legend(bbox_to_anchor=(0.5, 1.0), loc='lower center', ncol=3, borderaxespad=0.5)
                ax.set_ylabel(", ".join([format_measure_name(m) for m in pct_measures]))
            else:
                # Only value measures
                ax.legend(bbox_to_anchor=(0.5, 1.0), loc='lower center', ncol=3, borderaxespad=0.5)
                ax.set_ylabel(", ".join([format_measure_name(m) for m in value_measures]))
        elif color_column:
            # Single measure with color grouping
            for group in sorted_df[color_column].unique():
                group_data = sorted_df[sorted_df[color_column] == group]
                ax.plot(group_data[x_column], group_data[y_column], label=group, marker='o')
            ax.legend(bbox_to_anchor=(0.5, 1.0), loc='lower center', ncol=3, borderaxespad=0.5)
        else:
            # Simple line chart
            ax.plot(sorted_df[x_column], sorted_df[y_column], marker='o')
    elif chart_type == "scatter":
        sns.scatterplot(data=df, x=x_column, y=y_column, hue=color_column, ax=ax, s=60)
    elif chart_type == "heatmap":
        if y_column:
            pivot_df = df.pivot_table(index=x_column, columns=y_column, aggfunc='size', fill_value=0)

            # Sort heatmap: x_column ascending (index), y_column descending (columns)
            # Default behavior, can be overridden
            if not sort_by or sort_by == x_column:
                # Sort index (x_column) - default ascending
                x_sort_order = sort_order if sort_order else 'ascending'
                pivot_df = pivot_df.sort_index(ascending=(x_sort_order == 'ascending'))

            # Sort columns (y_column) descending by default
            if y_column:
                # For columns, we always want descending unless explicitly overridden
                pivot_df = pivot_df.sort_index(axis=1, ascending=False)
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns
            pivot_df = df[numeric_cols].corr()

            # Sort correlation heatmap indices
            if sort_order:
                pivot_df = pivot_df.sort_index(ascending=(sort_order == 'ascending'))
                pivot_df = pivot_df.sort_index(axis=1, ascending=(sort_order == 'ascending'))

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
    
    ax.set_xlabel(x_column)
    # Only set y_column as ylabel if it's not a multi-measure line chart (which sets its own labels)
    if y_column and not (chart_type == "line" and isinstance(y_column, list)):
        ax.set_ylabel(y_column)

    # Position any legend at top center for line charts with multiple measures, otherwise on the right
    legend = ax.get_legend()
    if legend is not None:
        if chart_type == "line" and isinstance(y_column, list):
            # Legend already positioned at top in the line chart code above
            # Add title with extra padding to make room for legend
            ax.set_title(title, fontsize=14, fontweight='bold', pad=40)
        else:
            ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
            ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(title, fontsize=14, fontweight='bold')

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