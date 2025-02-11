import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

@st.cache_data
def df_prepro(path):
    
    df = pd.read_parquet(path)
    df = df[['(Parent) ASIN', '(Child) ASIN', 'Title', 'SKU', 'Sessions',
           'Page Views', 'Buy Box Percentage',
           'Units Ordered',
           'Total Order Items',
           'Date']].copy()
    df.columns = df.columns.str.lower().str.replace(r'[()\-]', '', regex=True).str.replace(r'\s+', '_', regex=True)
    df = df[df.buy_box_percentage > 0]
    
    df['date'] = pd.to_datetime(df['date'])
    df['month_year'] = df['date'].dt.strftime('%B-%Y')

    # funnel(ayvax)
    df['sold'] = df.units_ordered / df.buy_box_percentage
    df['avg_unit_per_order'] = df.units_ordered / df.total_order_items
    df['order'] = df.sold / df.avg_unit_per_order
    df['sales_conversion_rate'] = df.order / df.sessions * 100

    # bar/line
    df['ordered_units'] = df.units_ordered / df.buy_box_percentage #-> duplicate with column 'sold'
    df['orders'] = df.total_order_items / df.buy_box_percentage
    df['avg_unit'] = df.ordered_units / df.orders
    df['conversion_rate_ou'] = (df.ordered_units / df.avg_unit) / df.sessions
    df['conversion_rate_o'] = (df.orders / df.avg_unit) / df.sessions
    
    # funnel (others)
    df['buy_box_percentage'] = df.buy_box_percentage * 100
    df['buy_box_percentage_others'] = 100 - df.buy_box_percentage
    df['units_ordered_others'] = df.buy_box_percentage_others / df.buy_box_percentage * df.units_ordered
    df['total_order_items_others'] = df.buy_box_percentage_others / df.buy_box_percentage * df.total_order_items
    
    return df.round(2)

    
def funnel_data(df):
    
    df = df[['sessions', 'page_views', 
            'units_ordered', 'units_ordered_others', 
            'total_order_items', 'total_order_items_others']].copy()
    
    m = df.select_dtypes(np.number)
    df[m.columns]= m.round().astype('Int64')
    df_vis = df.transpose().reset_index()
    df_vis = df_vis.rename(columns={df_vis.columns[0]: "stages", df_vis.columns[1]: "values"})
    df_vis['group'] = df_vis['stages'].apply(lambda x: group_stages(x))
    df_vis['stages'] = df_vis['stages'].str.replace('_others', '')
    df_vis['sort'] = df_vis.stages.apply(lambda x: order_cat(x))
    df_vis = df_vis.sort_values('sort')
    return df_vis

# Create custom color mapping
funnel_color_discrete_map = {
    'All': 'rgb(201, 219, 116)', # ligth green
    'Ayvax': 'rgb(201, 219, 116)',  # ligth green
    'Others': 'rgb(102, 166, 30)'   # dark green
}


def funnel_viz(df):

    fig = px.funnel(
        df,
        x='values',
        y='stages',
        color='group',
        color_discrete_map=funnel_color_discrete_map
    )

    return fig


def compare_funnel(df1, df2, asin1, asin2):

    
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=(f"Funnel for {asin1}", f"Funnel for {asin2}"),
        specs=[[{"type": "funnel"}, {"type": "funnel"}]]
    )
    
    
    for group_name in df1['group'].unique():
        group_df = df1[df1['group'] == group_name].sort_values('sort')
        fig.add_trace(
            go.Funnel(
                name=group_name,
                x=group_df['values'],
                y=group_df['stages'],
                marker=dict(color=funnel_color_discrete_map[group_name]),
                hovertemplate="<b>Group:</b> " + group_name + "<br><b>Stage:</b> %{y}<br><b>Values:</b> %{x}<extra></extra>"
            ),
            row=1, col=1
        )
    
    for group_name in df2['group'].unique():
        group_df = df2[df2['group'] == group_name].sort_values('sort')
        fig.add_trace(
            go.Funnel(
                name=group_name,
                x=group_df['values'],
                y=group_df['stages'],
                marker=dict(color=funnel_color_discrete_map[group_name]),
                hovertemplate="<b>Group:</b> " + group_name + "<br><b>Stage:</b> %{y}<br><b>Values:</b> %{x}<extra></extra>"
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        showlegend=False  #  disable the legend
    )

    return fig

def compare_funnel_2(df1, df2, asin1, asin2):
    st.write(df1)
    st.write(df2)
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=(f"Funnel for {asin1}", f"Funnel for {asin2}"),
        specs=[[{"type": "funnel"}, {"type": "funnel"}]]
    )
    
    
    fig1 = px.funnel(
        df1,
        x='values',
        y='stages',
        color='group',
        color_discrete_map=funnel_color_discrete_map)
    
    
    fig2 = px.funnel(
        df2,
        x='values',
        y='stages',
        color='group',
        color_discrete_map=funnel_color_discrete_map)
    
    fig1_traces, fig2_traces = [], []
    
    for trace in range(len(fig1["data"])):
        fig1_traces.append(fig1["data"][trace])
        
    for trace in range(len(fig2["data"])):
        fig2_traces.append(fig2["data"][trace])
        
    for traces in fig1_traces:
        fig.append_trace(traces, row=1, col=1)
    for traces in fig2_traces:
        fig.append_trace(traces, row=1, col=2)
    
    fig.update_layout(
        showlegend=False  #  disable the legend
    )

    return fig

def df_trend(df, asin, range):
    if range == 'Last 6 months':
        df = df[(df.date > (df.date.max() - pd.DateOffset(months=6))) & (df.parent_asin == asin)]
    elif range == 'Last 3 months':
        df = df[(df.date > (df.date.max() - pd.DateOffset(months=3))) & (df.parent_asin == asin)]

    return df

def group_stages(stage):
    if 'others' in stage:
        return 'Others'
    elif stage == 'sessions' or stage == 'page_views':
        return 'All'
    else:
        return 'Ayvax'

def order_cat(cat):
    if cat == 'total_order_items' or cat == 'orders':
        return 5
    elif cat == 'units_ordered'or cat == 'ordered_units':
        return 4
    elif cat == 'buy_box_percentage':
        return 3
    elif cat == 'sessions':
        return 2
    else : return 1

def bar_line_cluster(df):

    df_bar = df[['month_year', 'orders', 'ordered_units', 'sessions', 'page_views']].copy()
    
    ori_cols = df_bar.columns[1:]
    cols = [col + '_pct' for col in ori_cols]
    for index, col in enumerate(cols):
        if index == 3:
            df_bar[col] = 100
        else:
            df_bar[col] = ((df_bar[ori_cols[index]] / df_bar[ori_cols[index+1]]) * 100).round(2)

    df1 = df_bar.melt(
        id_vars=["month_year"], 
        value_vars= ori_cols,
        var_name='Category', 
        value_name='Total'
    ).reset_index()
    
    df2 = df_bar.melt(
        id_vars=["month_year"], 
        value_vars= cols,
        var_name='Category', 
        value_name='Pct'
    ).reset_index()

    df_bar = df1.merge(df2, on=['index', 'month_year'], how='left')
    df_bar['Total'] = df_bar['Total'].astype(int)

    df['date'] = pd.to_datetime(df['month_year'], format='%B-%Y')
    df = df.sort_values('date')

    df_bar['cat_order'] = df_bar.Category_x.apply(lambda x: order_cat(x))
    df_bar['date'] = pd.to_datetime(df_bar['month_year'], format='%B-%Y')
    df_bar = df_bar.sort_values(['date', 'cat_order'])

    fig = px.bar(df_bar, 
                 x='month_year', 
                 y='Total', 
                 color='Category_x', 
                 hover_data={'Pct': ':.2f'},  
                 labels={'Pct': 'Percentage', 'month_year': 'Date'},
                 barmode='group')

    fig.update_traces(hovertemplate='Total: %{y}<br>Percentage: %{customdata[0]}%')

    # df['conversion_rate_pct'] = df.conversion_rate_adj * 100
    
    fig.add_trace(go.Scatter(
        x=df['month_year'],
        y=df['conversion_rate_ou'],  
        mode='lines+markers',
        name='Conversion Rate (Order Units)',
        line=dict(color='green', width=3),
        yaxis='y2'
    ))

    fig.add_trace(go.Scatter(
            x=df['month_year'],
            y=df['conversion_rate_o'],  
            mode='lines+markers',
            name='Conversion Rate (Orders)',
            line=dict(color='yellow', width=3),
            yaxis='y2'
        ))
        
    fig.update_layout(
        xaxis=dict(title='Date'),
        yaxis=dict(
            title='Total',
            side='left',
            showgrid=False
        ),
        yaxis2=dict(
            overlaying='y',
            side='right',
            range=[0, 1],  # Batas skala 0-1
            showgrid=False#,
            #tickformat=',.0%'  # Format persentase
        ),
        barmode='group',  # Bar dikelompokkan
        legend=dict(x=1.1, y=1)  # Posisi legenda
    )
    return fig

def bar_line_stacked(df):

    cols = ['page_views', 'sessions', 'ordered_units', 'orders', 'conversion_rate_ou', 'conversion_rate_o']
    stages = ['Page Views', 'Sessions', 'Order Units', 'Orders', 'Conversion Rate (Order Units)', 'Conversion Rate (Orders)']
    colors = ['royalblue', 'lightblue', 'pink', 'red', 'yellow', 'green']
    
    fig = go.Figure()
    for i in range(len(cols)):
        if i < 4:
            fig.add_trace(go.Bar(
                x=df['month_year'],
                y=df[cols[i]],
                name=stages[i],
                marker_color=colors[i],
                offsetgroup=1  # for grouping bar
            ))
        else:
            fig.add_trace(go.Scatter(
                x=df['month_year'],
                y=df[cols[i]],
                name=stages[i],
                mode='lines+markers',
                line=dict(color=colors[i], width=3),
                yaxis='y2'  # using second y-axis
            ))
            
    fig.update_layout(
        title='Sales PPC & Conversion Trend',
        #xaxis=dict(title='Date'),
        yaxis=dict(
            #title='Total',
            side='left',
            showgrid=False
        ),
        yaxis2=dict(
            #title='Conversion Rate',
            overlaying='y',
            side='right',
            range=[0, 1],  
            showgrid=False#,
           # tickformat=',.0%'  # pct format
        ),
        legend=dict(x=1.1, y=1)  # legend position
    )
    
    return fig
    