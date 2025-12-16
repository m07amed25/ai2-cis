import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'danger': '#D64933',
    'info': '#6C63FF',
    'warning': '#FFB703',
    'dark': '#2B2D42',
    'light': '#EDF2F4',
    'gradient': ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D64933']
}

DARK_LAYOUT = {
    'template': 'plotly_dark',
    'plot_bgcolor': '#1a1f2e',
    'paper_bgcolor': '#1a1f2e',
    'font': {'color': '#e8eaed'}
}

st.set_page_config(
    page_title="California Housing EDA",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #000000;
    }
    
    .block-container {
        background-color: #0a0e1a;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.8);
    }
    
    .stMarkdown, .stDataFrame, div[data-testid="stVerticalBlock"] {
        background-color: transparent;
    }
    
    .main * {
        color: #f0f0f0;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    p, li, span, label {
        color: #c5c5c5 !important;
    }
    
    .stTextInput input, .stSelectbox select, .stMultiSelect {
        background-color: #1a1f2e !important;
        color: #f0f0f0 !important;
        border: 1px solid #2d3748 !important;
    }
    
    .stNumberInput input {
        background-color: #1a1f2e !important;
        color: #f0f0f0 !important;
        border: 1px solid #2d3748 !important;
    }
    
    .stSlider {
        background-color: transparent !important;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #4a5bd6 0%, #5a3a7a 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        color: white !important;
    }
    
    .stMetric label {
        color: rgba(255,255,255,0.95) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1f2e;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        background-color: transparent;
        border: none;
        color: #a0a0a0 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4a5bd6 0%, #5a3a7a 100%);
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #0a0e1a;
        padding: 20px;
        border-radius: 10px;
    }
    
    h1 {
        color: #ffffff !important;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e1a 0%, #000000 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #f0f0f0 !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: #1a1f2e !important;
    }
    
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 10px rgba(0,0,0,0.5);
        background-color: #1a1f2e !important;
    }
    
    .stAlert * {
        color: #f0f0f0 !important;
    }
    
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        background-color: #1a1f2e !important;
    }
    
    .dataframe * {
        background-color: #1a1f2e !important;
        color: #f0f0f0 !important;
    }
    
    .dataframe thead tr th {
        background-color: #0a0e1a !important;
        color: #ffffff !important;
    }
    
    .js-plotly-plot {
        background-color: #0a0e1a !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #4a5bd6 0%, #5a3a7a 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(74, 91, 214, 0.5);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(74, 91, 214, 0.7);
        transform: translateY(-2px);
    }
    
    .stFileUploader {
        background-color: #1a1f2e;
        border: 2px dashed #4a5bd6;
        border-radius: 15px;
        padding: 20px;
    }
    
    .stFileUploader * {
        color: #f0f0f0 !important;
    }
    
    div[data-testid="stExpander"] {
        background-color: #1a1f2e;
        border: 1px solid #2d3748;
    }
    
    .stDownloadButton button {
        background: linear-gradient(135deg, #4a5bd6 0%, #5a3a7a 100%);
        color: white !important;
        border: none;
    }
    
    input {
        background-color: #1a1f2e !important;
        color: #f0f0f0 !important;
    }
    
    [data-baseweb="select"] {
        background-color: #1a1f2e !important;
    }
    
    [data-baseweb="select"] * {
        background-color: #1a1f2e !important;
        color: #f0f0f0 !important;
    }
    
    .stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput {
        color: #f0f0f0 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center; font-size: 3rem; margin-bottom: 0;'>
        California Housing Analytics
    </h1>
    <p style='text-align: center; color: #b8bcc4; font-size: 1.2rem; margin-top: 0;'>
        Exploratory Data Analysis & Insights
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload your housing dataset (CSV format)", 
    type=['csv'],
    help="Upload your California housing dataset to begin analysis"
)

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        return df
    
    df = load_data(uploaded_file)
    
    st.sidebar.markdown("## Data Filters")
    st.sidebar.markdown("---")
    
    ocean_filter = st.sidebar.multiselect(
        "Ocean Proximity",
        options=sorted(df['ocean_proximity'].unique()),
        default=df['ocean_proximity'].unique(),
        help="Filter by distance to ocean"
    )
    
    price_min, price_max = st.sidebar.slider(
        "House Value Range ($)",
        int(df['median_house_value'].min()),
        int(df['median_house_value'].max()),
        (int(df['median_house_value'].min()), int(df['median_house_value'].max())),
        format="$%d"
    )
    
    income_min, income_max = st.sidebar.slider(
        "Median Income Range (10k$)",
        float(df['median_income'].min()),
        float(df['median_income'].max()),
        (float(df['median_income'].min()), float(df['median_income'].max())),
        step=0.1
    )
    
    age_min, age_max = st.sidebar.slider(
        "Housing Age Range (years)",
        int(df['housing_median_age'].min()),
        int(df['housing_median_age'].max()),
        (int(df['housing_median_age'].min()), int(df['housing_median_age'].max()))
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Display Options")
    show_outliers = st.sidebar.checkbox("Show Outliers", value=True)
    sample_size = st.sidebar.slider("Sample Size for Plots", 1000, 10000, 3000, 500)
    
    df_filtered = df[
        (df['ocean_proximity'].isin(ocean_filter)) &
        (df['median_house_value'] >= price_min) &
        (df['median_house_value'] <= price_max) &
        (df['median_income'] >= income_min) &
        (df['median_income'] <= income_max) &
        (df['housing_median_age'] >= age_min) &
        (df['housing_median_age'] <= age_max)
    ]
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Overview", 
        "Distributions",
        "Correlations",
        "Geographic",
        "Feature Engineering",
        "Statistical Tests",
        "Price Prediction",
    ])
    
    with tab1:
        st.markdown("## Dataset Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Records", 
                f"{len(df_filtered):,}",
                delta=f"{len(df_filtered)-len(df):,}" if len(df_filtered) != len(df) else None
            )
        
        with col2:
            avg_value = df_filtered['median_house_value'].mean()
            st.metric(
                "Avg House Value", 
                f"${avg_value:,.0f}",
                delta=f"{((avg_value/df['median_house_value'].mean()-1)*100):.1f}%" if len(df_filtered) != len(df) else None
            )
        
        with col3:
            avg_income = df_filtered['median_income'].mean()
            st.metric(
                "Avg Income", 
                f"${avg_income * 10000:,.0f}",
                delta=f"{((avg_income/df['median_income'].mean()-1)*100):.1f}%" if len(df_filtered) != len(df) else None
            )
        
        with col4:
            st.metric(
                "Median Age", 
                f"{df_filtered['housing_median_age'].median():.0f} yrs"
            )
        
        with col5:
            missing = df_filtered['total_bedrooms'].isna().sum()
            missing_pct = (missing / len(df_filtered) * 100)
            st.metric(
                "Missing Values", 
                f"{missing:,}",
                delta=f"{missing_pct:.1f}%",
                delta_color="inverse"
            )
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### - Descriptive Statistics")
            
            stats_df = df_filtered.describe().T
            stats_df['missing'] = df_filtered.isna().sum()
            stats_df['missing_pct'] = (stats_df['missing'] / len(df_filtered) * 100).round(2)
            stats_df['skewness'] = df_filtered.select_dtypes(include=[np.number]).skew()
            stats_df['kurtosis'] = df_filtered.select_dtypes(include=[np.number]).kurtosis()
            
            def color_missing(val):
                if val > 5:
                    return 'background-color: #ffcccc'
                elif val > 1:
                    return 'background-color: #ffffcc'
                return ''
            
            styled_df = stats_df.style.format({
                'mean': '{:.2f}',
                'std': '{:.2f}',
                'min': '{:.2f}',
                '25%': '{:.2f}',
                '50%': '{:.2f}',
                '75%': '{:.2f}',
                'max': '{:.2f}',
                'missing_pct': '{:.2f}%',
                'skewness': '{:.2f}',
                'kurtosis': '{:.2f}'
            }).applymap(color_missing, subset=['missing_pct'])
            
            st.dataframe(styled_df, use_container_width=True, height=400)
        
        with col2:
            st.markdown("### Data Quality Score")
            
            # data quality score
            completeness = (1 - df_filtered.isna().sum().sum() / (len(df_filtered) * len(df_filtered.columns))) * 100
            consistency = 100 - (df_filtered.duplicated().sum() / len(df_filtered) * 100)
            
            quality_score = (completeness + consistency) / 2
            
            # gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=quality_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Quality Score", 'font': {'size': 24, 'color': '#e8eaed'}},
                delta={'reference': 95, 'increasing': {'color': COLORS['success']}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': '#e8eaed'},
                    'bar': {'color': COLORS['primary']},
                    'bgcolor': "#1a1f2e",
                    'borderwidth': 2,
                    'bordercolor': '#4a5568',
                    'steps': [
                        {'range': [0, 50], 'color': COLORS['danger']},
                        {'range': [50, 80], 'color': COLORS['warning']},
                        {'range': [80, 100], 'color': COLORS['success']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(
                height=250, 
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='#1a1f2e',
                font={'color': '#e8eaed'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Quality metrics
            st.markdown("#### Quality Metrics")
            st.metric("Completeness", f"{completeness:.1f}%")
            st.metric("Consistency", f"{consistency:.1f}%")
            st.metric("Duplicates", f"{df_filtered.duplicated().sum()}")
        
        st.markdown("---")
        
        # preview with search
        st.markdown("### Data Preview")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("Search in data", "")
        with col2:
            num_rows = st.selectbox("Rows to display", [5, 10, 20, 50], index=1)
        
        if search_term:
            mask = df_filtered.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
            st.dataframe(df_filtered[mask].head(num_rows), use_container_width=True)
        else:
            st.dataframe(df_filtered.head(num_rows), use_container_width=True)
    
    with tab2:
        st.markdown("## - Feature Distributions")
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df_filtered['median_house_value'],
                nbinsx=50,
                name='Distribution',
                marker_color=COLORS['primary'],
                opacity=0.7
            ))
            
            fig.update_layout(
                title='House Value Distribution',
                xaxis_title='House Value ($)',
                yaxis_title='Frequency',
                template='plotly_dark',
                height=400,
                showlegend=True,
                plot_bgcolor='#1a1f2e',
                paper_bgcolor='#1a1f2e',
                font=dict(color='#e8eaed')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df_filtered['median_income'],
                nbinsx=50,
                name='Distribution',
                marker_color=COLORS['success'],
                opacity=0.7
            ))
            
            fig.update_layout(
                title='Median Income Distribution',
                xaxis_title='Median Income (10k$)',
                yaxis_title='Frequency',
                template='plotly_white',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df_filtered['housing_median_age'],
                nbinsx=30,
                name='Distribution',
                marker_color=COLORS['accent'],
                opacity=0.7
            ))
            
            fig.update_layout(
                title='Housing Age Distribution',
                xaxis_title='Housing Age (years)',
                yaxis_title='Frequency',
                template='plotly_white',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            ocean_counts = df_filtered['ocean_proximity'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=ocean_counts.index,
                values=ocean_counts.values,
                hole=0.4,
                marker=dict(colors=COLORS['gradient']),
                textposition='inside',
                textinfo='percent+label'
            )])
            
            fig.update_layout(
                title='Ocean Proximity Distribution',
                template='plotly_white',
                height=400,
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### - Box Plots - Outlier Detection")
        
        numeric_cols = ['total_rooms', 'total_bedrooms', 'population', 'households']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=numeric_cols
        )
        
        colors_box = [COLORS['primary'], COLORS['success'], COLORS['accent'], COLORS['secondary']]
        
        for idx, col in enumerate(numeric_cols):
            row = idx // 2 + 1
            col_num = idx % 2 + 1
            
            fig.add_trace(
                go.Box(
                    y=df_filtered[col],
                    name=col,
                    marker_color=colors_box[idx],
                    boxmean='sd'
                ),
                row=row, col=col_num
            )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            template='plotly_white',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### - Violin Plots - Distribution Shape")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            for proximity in df_filtered['ocean_proximity'].unique():
                fig.add_trace(go.Violin(
                    y=df_filtered[df_filtered['ocean_proximity'] == proximity]['median_house_value'],
                    name=proximity,
                    box_visible=True,
                    meanline_visible=True
                ))
            
            fig.update_layout(
                title='House Value by Ocean Proximity',
                yaxis_title='House Value ($)',
                template='plotly_white',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            for proximity in df_filtered['ocean_proximity'].unique():
                fig.add_trace(go.Violin(
                    y=df_filtered[df_filtered['ocean_proximity'] == proximity]['median_income'],
                    name=proximity,
                    box_visible=True,
                    meanline_visible=True
                ))
            
            fig.update_layout(
                title='Income by Ocean Proximity',
                yaxis_title='Median Income (10k$)',
                template='plotly_white',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("## - Correlation Analysis")
        
        numeric_df = df_filtered.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title='Correlation Matrix Heatmap',
            template='plotly_white',
            height=700,
            xaxis={'side': 'bottom'},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Interactive scatter plots
        st.markdown("### - Feature Relationships")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("Select X variable", numeric_df.columns, index=7)
        with col2:
            y_var = st.selectbox("Select Y variable", numeric_df.columns, index=8)
        
        sample_data = df_filtered.sample(min(sample_size, len(df_filtered)))
        
        fig = px.scatter(
            sample_data,
            x=x_var,
            y=y_var,
            color='ocean_proximity',
            size='population',
            hover_data=['housing_median_age', 'median_income'],
            title=f'{x_var} vs {y_var}',
            color_discrete_sequence=COLORS['gradient'],
            opacity=0.6
        )
        
        fig.update_layout(
            template='plotly_white',
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### - Top Correlations with House Value")
        
        target_corr = corr_matrix['median_house_value'].sort_values(ascending=False)
        target_corr_df = pd.DataFrame({
            'Feature': target_corr.index,
            'Correlation': target_corr.values
        }).query('Feature != "median_house_value"')
        
        fig = go.Figure()
        
        colors = [COLORS['success'] if x > 0 else COLORS['danger'] for x in target_corr_df['Correlation']]
        
        fig.add_trace(go.Bar(
            y=target_corr_df['Feature'],
            x=target_corr_df['Correlation'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=target_corr_df['Correlation'].round(3),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Features Correlation with House Value',
            xaxis_title='Correlation Coefficient',
            yaxis_title='Feature',
            template='plotly_white',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### - Pair Plot Analysis")
        
        selected_features = st.multiselect(
            "Select features for pair plot",
            numeric_df.columns.tolist(),
            default=['median_income', 'median_house_value', 'housing_median_age']
        )
        
        if len(selected_features) >= 2:
            sample_pair = df_filtered[selected_features].sample(min(1000, len(df_filtered)))
            fig = px.scatter_matrix(
                sample_pair,
                dimensions=selected_features,
                color_discrete_sequence=[COLORS['primary']],
                opacity=0.5
            )
            fig.update_layout(
                height=800,
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("## - Geographic Analysis")
        
        sample_geo = df_filtered.sample(min(sample_size, len(df_filtered)))
        
        # map
        fig = px.scatter_mapbox(
            sample_geo,
            lat='latitude',
            lon='longitude',
            color='median_house_value',
            size='population',
            hover_data={
                'median_income': ':.2f',
                'housing_median_age': ':.0f',
                'ocean_proximity': True,
                'median_house_value': ':$,.0f',
                'latitude': ':.2f',
                'longitude': ':.2f'
            },
            color_continuous_scale='Viridis',
            title=f'Geographic Distribution (Sample: {len(sample_geo):,} points)',
            zoom=5,
            height=700,
            size_max=15
        )
        
        fig.update_layout(
            mapbox_style="carto-positron",
            margin=dict(l=0, r=0, t=30, b=0),
            coloraxis_colorbar=dict(title="House Value")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Geographic patterns
        col1, col2 = st.columns(2)
        
        with col1:
            # Density heatmap
            fig = go.Figure(go.Histogram2d(
                x=sample_geo['longitude'],
                y=sample_geo['latitude'],
                colorscale='Viridis',
                nbinsx=50,
                nbinsy=50
            ))
            fig.update_layout(
                title='Population Density Heatmap',
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            ocean_stats = df_filtered.groupby('ocean_proximity')['median_house_value'].agg(['mean', 'median', 'std']).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ocean_stats['ocean_proximity'],
                y=ocean_stats['mean'],
                name='Mean',
                marker_color=COLORS['primary'],
                error_y=dict(type='data', array=ocean_stats['std'])
            ))
            fig.add_trace(go.Bar(
                x=ocean_stats['ocean_proximity'],
                y=ocean_stats['median'],
                name='Median',
                marker_color=COLORS['success']
            ))
            
            fig.update_layout(
                title='House Value Statistics by Ocean Proximity',
                xaxis_title='Ocean Proximity',
                yaxis_title='House Value ($)',
                template='plotly_white',
                height=500,
                barmode='group',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("## - Feature Engineering")
        
        df_fe = df_filtered.copy()
        df_fe['rooms_per_household'] = df_fe['total_rooms'] / df_fe['households']
        df_fe['bedrooms_per_room'] = df_fe['total_bedrooms'] / df_fe['total_rooms']
        df_fe['population_per_household'] = df_fe['population'] / df_fe['households']
        df_fe['bedroom_ratio'] = df_fe['total_bedrooms'] / df_fe['total_rooms']
        
        df_fe = df_fe.replace([np.inf, -np.inf], np.nan)
        
        st.markdown("### Engineered Features Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Rooms/Household",
                f"{df_fe['rooms_per_household'].mean():.2f}",
                delta=f"±{df_fe['rooms_per_household'].std():.2f}"
            )
        
        with col2:
            st.metric(
                "Bedrooms/Room",
                f"{df_fe['bedrooms_per_room'].mean():.2f}",
                delta=f"±{df_fe['bedrooms_per_room'].std():.2f}"
            )
        
        with col3:
            st.metric(
                "Population/Household",
                f"{df_fe['population_per_household'].mean():.2f}",
                delta=f"±{df_fe['population_per_household'].std():.2f}"
            )
        
        with col4:
            st.metric(
                "Bedroom Ratio",
                f"{df_fe['bedroom_ratio'].mean():.2f}",
                delta=f"±{df_fe['bedroom_ratio'].std():.2f}"
            )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df_fe['rooms_per_household'].dropna(),
                nbinsx=50,
                name='Rooms per Household',
                marker_color=COLORS['primary'],
                opacity=0.7
            ))
            
            fig.update_layout(
                title='Rooms per Household Distribution',
                xaxis_title='Rooms per Household',
                yaxis_title='Frequency',
                template='plotly_white',
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            sample_fe = df_fe.sample(min(2000, len(df_fe)))
            fig = px.scatter(
                sample_fe,
                x='rooms_per_household',
                y='median_house_value',
                color='ocean_proximity',
                title='Rooms per Household vs House Value',
                opacity=0.6,
                color_discrete_sequence=COLORS['gradient']
            )
            fig.update_layout(
                template='plotly_white',
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df_fe['population_per_household'].dropna(),
                nbinsx=50,
                name='Population per Household',
                marker_color=COLORS['success'],
                opacity=0.7
            ))
            
            fig.update_layout(
                title='Population per Household Distribution',
                xaxis_title='Population per Household',
                yaxis_title='Frequency',
                template='plotly_white',
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            fig = px.scatter(
                sample_fe,
                x='population_per_household',
                y='median_house_value',
                color='ocean_proximity',
                title='Population per Household vs House Value',
                opacity=0.6,
                color_discrete_sequence=COLORS['gradient']
            )
            fig.update_layout(
                template='plotly_white',
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### - Engineered Features Correlation")
        
        new_features = ['rooms_per_household', 'bedrooms_per_room', 'population_per_household', 'bedroom_ratio']
        fe_corr = df_fe[new_features + ['median_house_value']].corr()['median_house_value'].drop('median_house_value').sort_values(ascending=False)
        
        fig = go.Figure()
        
        colors = [COLORS['success'] if x > 0 else COLORS['danger'] for x in fe_corr.values]
        
        fig.add_trace(go.Bar(
            x=fe_corr.values,
            y=fe_corr.index,
            orientation='h',
            marker=dict(color=colors),
            text=fe_corr.values.round(3),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Engineered Features Correlation with House Value',
            xaxis_title='Correlation Coefficient',
            yaxis_title='Feature',
            template='plotly_white',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### - Feature Importance Comparison")
        
        all_corr = df_fe.select_dtypes(include=[np.number]).corr()['median_house_value'].drop('median_house_value').sort_values(ascending=False).head(10)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=all_corr.index,
            x=all_corr.values,
            orientation='h',
            marker=dict(
                color=all_corr.values,
                colorscale='RdYlGn',
                showscale=True
            ),
            text=all_corr.values.round(3),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Top 10 Features Correlation with House Value',
            xaxis_title='Correlation Coefficient',
            yaxis_title='Feature',
            template='plotly_white',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.markdown("## - Statistical Tests")
        
        st.markdown("### Normality Tests")
        
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        
        normality_results = []
        for col in numeric_cols:
            data_clean = df_filtered[col].dropna()
            if len(data_clean) > 0:
                stat, p_value = stats.shapiro(data_clean.sample(min(5000, len(data_clean))))
                normality_results.append({
                    'Feature': col,
                    'Shapiro-Wilk Statistic': stat,
                    'P-Value': p_value,
                    'Normal': 'Yes' if p_value > 0.05 else 'No'
                })
        
        normality_df = pd.DataFrame(normality_results)
        
        def highlight_normal(row):
            if row['Normal'] == 'Yes':
                return ['background-color: #d4edda'] * len(row)
            else:
                return ['background-color: #f8d7da'] * len(row)
        
        styled_normality = normality_df.style.apply(highlight_normal, axis=1).format({
            'Shapiro-Wilk Statistic': '{:.4f}',
            'P-Value': '{:.4f}'
        })
        
        st.dataframe(styled_normality, use_container_width=True)
        
        st.info("💡 P-value > 0.05 suggests the data is normally distributed")
        
        st.markdown("---")
        
        # Q-Q plots
        st.markdown("### - Q-Q Plots (Quantile-Quantile)")
        
        selected_col = st.selectbox("Select feature for Q-Q plot", numeric_cols)
        
        data_qq = df_filtered[selected_col].dropna()
        
        fig = go.Figure()
        
        # theoretical quantiles
        sorted_data = np.sort(data_qq)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
        
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sorted_data,
            mode='markers',
            marker=dict(color=COLORS['primary'], size=5, opacity=0.6),
            name='Data'
        ))
        
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=theoretical_quantiles * np.std(sorted_data) + np.mean(sorted_data),
            mode='lines',
            line=dict(color=COLORS['danger'], width=2, dash='dash'),
            name='Normal Distribution'
        ))
        
        fig.update_layout(
            title=f'Q-Q Plot for {selected_col}',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles',
            template='plotly_white',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### - Hypothesis Testing")
        
        st.markdown("#### Test: Does ocean proximity affect house values?")
        
        # ANOVA test
        groups = [df_filtered[df_filtered['ocean_proximity'] == cat]['median_house_value'].dropna() 
                  for cat in df_filtered['ocean_proximity'].unique()]
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("F-Statistic", f"{f_stat:.4f}")
        with col2:
            st.metric("P-Value", f"{p_value:.6f}")
        
        if p_value < 0.05:
            st.success("✅ Reject null hypothesis: Ocean proximity significantly affects house values (p < 0.05)")
        else:
            st.warning("⚠️ Cannot reject null hypothesis: No significant effect detected (p ≥ 0.05)")
        
        st.markdown("---")
        
        st.markdown("### - Correlation Significance Tests")
        
        corr_sig_results = []
        target = df_filtered['median_house_value'].dropna()
        
        for col in numeric_cols:
            if col != 'median_house_value':
                feature = df_filtered[col].dropna()
                common_idx = target.index.intersection(feature.index)
                
                if len(common_idx) > 2:
                    corr_coef, p_val = stats.pearsonr(target[common_idx], feature[common_idx])
                    corr_sig_results.append({
                        'Feature': col,
                        'Correlation': corr_coef,
                        'P-Value': p_val,
                        'Significant': 'Yes' if p_val < 0.05 else 'No'
                    })
        
        corr_sig_df = pd.DataFrame(corr_sig_results).sort_values('Correlation', ascending=False)
        
        def highlight_significant(row):
            if row['Significant'] == 'Yes':
                return ['background-color: #d4edda'] * len(row)
            else:
                return ['background-color: #fff3cd'] * len(row)
        
        styled_corr_sig = corr_sig_df.style.apply(highlight_significant, axis=1).format({
            'Correlation': '{:.4f}',
            'P-Value': '{:.6f}'
        })
        
        st.dataframe(styled_corr_sig, use_container_width=True)
        
        st.info("💡 P-value < 0.05 indicates statistically significant correlation")
    
    with tab7:
        st.markdown("## House Price Prediction")
        
        model_path = r'D:\AI\ai2-project\models\best_model.pkl'
        scaler_path = r'D:\AI\ai2-project\models\scaler.pkl'
        metadata_path = r'D:\AI\ai2-project\models\model_metadata.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(metadata_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            st.success(f"- **Loaded Model**: {metadata['model_name']}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test RMSE", f"${metadata['test_rmse']:,.2f}")
            with col2:
                st.metric("Test MAE", f"${metadata['test_mae']:,.2f}")
            with col3:
                st.metric("R² Score", f"{metadata['test_r2']:.4f}")
            with col4:
                st.metric("CV RMSE", f"${metadata['cv_rmse']:,.2f}")
            
            st.markdown("---")
            
            st.markdown("### Enter Property Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Location")
                longitude = st.number_input(
                    "Longitude",
                    min_value=-124.5,
                    max_value=-114.0,
                    value=-118.0,
                    step=0.1,
                    help="Geographic longitude coordinate"
                )
                
                latitude = st.number_input(
                    "Latitude",
                    min_value=32.5,
                    max_value=42.0,
                    value=34.0,
                    step=0.1,
                    help="Geographic latitude coordinate"
                )
                
                ocean_proximity = st.selectbox(
                    "Ocean Proximity",
                    options=['<1H OCEAN', 'INLAND', 'NEAR BAY', 'NEAR OCEAN', 'ISLAND'],
                    help="Distance to ocean"
                )
            
            with col2:
                st.markdown("#### Property Info")
                housing_median_age = st.slider(
                    "Housing Age (years)",
                    min_value=1,
                    max_value=52,
                    value=25,
                    help="Median age of houses in the area"
                )
                
                total_rooms = st.number_input(
                    "Total Rooms",
                    min_value=1,
                    max_value=10000,
                    value=2000,
                    step=100,
                    help="Total number of rooms in the area"
                )
                
                total_bedrooms = st.number_input(
                    "Total Bedrooms",
                    min_value=1,
                    max_value=5000,
                    value=400,
                    step=50,
                    help="Total number of bedrooms in the area"
                )
            
            with col3:
                st.markdown("#### Demographics")
                population = st.number_input(
                    "Population",
                    min_value=1,
                    max_value=10000,
                    value=1000,
                    step=100,
                    help="Population in the area"
                )
                
                households = st.number_input(
                    "Households",
                    min_value=1,
                    max_value=5000,
                    value=400,
                    step=50,
                    help="Number of households in the area"
                )
                
                median_income = st.number_input(
                    "Median Income (in $10k)",
                    min_value=0.5,
                    max_value=15.0,
                    value=3.5,
                    step=0.1,
                    help="Median income in units of $10,000"
                )
            
            st.markdown("---")
            
            if st.button("Predict House Price", type="primary", use_container_width=True):
                input_data = {
                    'longitude': longitude,
                    'latitude': latitude,
                    'housing_median_age': housing_median_age,
                    'total_rooms': total_rooms,
                    'total_bedrooms': total_bedrooms,
                    'population': population,
                    'households': households,
                    'median_income': median_income
                }
                
                # engineered features
                input_data['rooms_per_household'] = total_rooms / households
                input_data['bedrooms_per_room'] = total_bedrooms / total_rooms
                input_data['population_per_household'] = population / households
                input_data['bedrooms_per_household'] = total_bedrooms / households
                input_data['income_squared'] = median_income ** 2
                input_data['income_rooms_interaction'] = median_income * input_data['rooms_per_household']
                input_data['is_new'] = 1 if housing_median_age < 10 else 0
                input_data['is_old'] = 1 if housing_median_age > 40 else 0
                input_data['people_per_room'] = population / total_rooms
                input_data['rooms_density'] = total_rooms / households
                
                # One-hot encode ocean_proximity
                for prox in ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']:
                    input_data[f'ocean_proximity_{prox}'] = 1 if ocean_proximity == prox else 0
                
                # correct feature order
                input_df = pd.DataFrame([input_data])
                
                # Reordering
                input_df = input_df[metadata['features']]
                
                input_scaled = scaler.transform(input_df)
                
                prediction = model.predict(input_scaled)[0]
                
                st.markdown("### Prediction Result")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
                        <h2 style='color: white !important; margin: 0; font-family: Arial, sans-serif;'>Predicted House Value</h2>
                        <h1 style='color: #ffffff !important; font-size: 3.5rem; margin: 20px 0; font-weight: 700; font-family: Arial, sans-serif;'>${prediction:,.0f}</h1>
                        <p style='color: rgba(255,255,255,0.9) !important; margin: 0; font-family: Arial, sans-serif;'>Estimated Market Value</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                lower_bound = prediction - metadata['test_mae']
                upper_bound = prediction + metadata['test_mae']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Lower Estimate",
                        f"${lower_bound:,.0f}",
                        delta=f"-${metadata['test_mae']:,.0f}"
                    )
                
                with col2:
                    st.metric(
                        "Predicted Value",
                        f"${prediction:,.0f}",
                        delta="Best Estimate"
                    )
                
                with col3:
                    st.metric(
                        "Upper Estimate",
                        f"${upper_bound:,.0f}",
                        delta=f"+${metadata['test_mae']:,.0f}"
                    )
                
                st.info(f"""
                 **About this prediction:**
                - Model accuracy: R² = {metadata['test_r2']:.4f}
                - Average error: ±${metadata['test_mae']:,.0f}
                - Confidence range: ${lower_bound:,.0f} - ${upper_bound:,.0f}
                - This prediction is based on {metadata['model_name']}
                """)
                
                # Feature importance for this prediction
                st.markdown("### Input Feature Analysis")
                
                feature_values = pd.DataFrame({
                    'Feature': list(input_data.keys())[:8],
                    'Value': [input_data[k] for k in list(input_data.keys())[:8]]
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Original Features")
                    st.dataframe(feature_values, use_container_width=True, height=300)
                
                with col2:
                    st.markdown("#### Engineered Features")
                    engineered_features = pd.DataFrame({
                        'Feature': [
                            'Rooms per Household',
                            'Bedrooms per Room',
                            'Population per Household',
                            'Income Squared',
                            'Is New Building',
                            'Is Old Building'
                        ],
                        'Value': [
                            f"{input_data['rooms_per_household']:.2f}",
                            f"{input_data['bedrooms_per_room']:.2f}",
                            f"{input_data['population_per_household']:.2f}",
                            f"{input_data['income_squared']:.2f}",
                            "Yes" if input_data['is_new'] == 1 else "No",
                            "Yes" if input_data['is_old'] == 1 else "No"
                        ]
                    })
                    st.dataframe(engineered_features, use_container_width=True, height=300)
                
                # Comparison with dataset
                if len(df_filtered) > 0:
                    st.markdown("---")
                    st.markdown("###Comparison with Dataset")
                    
                    similar_mask = (
                        (df_filtered['ocean_proximity'] == ocean_proximity) &
                        (df_filtered['median_income'].between(median_income - 1, median_income + 1))
                    )
                    
                    similar_properties = df_filtered[similar_mask]
                    
                    if len(similar_properties) > 0:
                        avg_similar = similar_properties['median_house_value'].mean()
                        min_similar = similar_properties['median_house_value'].min()
                        max_similar = similar_properties['median_house_value'].max()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Similar Properties",
                                f"{len(similar_properties):,}",
                                delta="Found in dataset"
                            )
                        
                        with col2:
                            st.metric(
                                "Avg Value (Similar)",
                                f"${avg_similar:,.0f}",
                                delta=f"{((prediction/avg_similar - 1) * 100):.1f}%"
                            )
                        
                        with col3:
                            st.metric(
                                "Min Value (Similar)",
                                f"${min_similar:,.0f}"
                            )
                        
                        with col4:
                            st.metric(
                                "Max Value (Similar)",
                                f"${max_similar:,.0f}"
                            )
                        
                        # Distribution comparison
                        fig = go.Figure()
                        
                        fig.add_trace(go.Histogram(
                            x=similar_properties['median_house_value'],
                            name='Similar Properties',
                            nbinsx=30,
                            marker_color=COLORS['primary'],
                            opacity=0.7
                        ))
                        
                        fig.add_vline(
                            x=prediction,
                            line_dash="dash",
                            line_color=COLORS['danger'],
                            line_width=3,
                            annotation_text=f"Your Prediction: ${prediction:,.0f}",
                            annotation_position="top"
                        )
                        
                        fig.update_layout(
                            title='Your Prediction vs Similar Properties',
                            xaxis_title='House Value ($)',
                            yaxis_title='Frequency',
                            template='plotly_dark',
                            height=400,
                            plot_bgcolor='#1a1f2e',
                            paper_bgcolor='#1a1f2e',
                            font=dict(color='#e8eaed')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No similar properties found in the dataset for comparison.")
        
        else:
            st.error("**Model Not Found!**")
            st.warning("""
            The trained model files are missing. Please ensure you have:
            
            1. Run the `model.ipynb` notebook to train the model
            2. Model files saved in: `D:\\AI\\ai2-project\\models\\`
            3. Required files:
               - `best_model.pkl`
               - `scaler.pkl`
               - `model_metadata.pkl`
            
            After training the model, refresh this page to enable predictions.
            """)
            
            st.info("**Tip**: Open the `notebooks/model.ipynb` file and run all cells to train and save the model.")
    
else:
    st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2>Welcome to California Housing Analytics! </h2>
            <p style='font-size: 1.2rem; color: #666; margin-top: 20px;'>
                Upload your housing dataset to unlock powerful insights and analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
    
    with col2:
        st.markdown("""
        ### - Expected Data Format:
        
        Your CSV should contain these columns:
        
        | Column | Type | Description |
        |--------|------|-------------|
        | `longitude` | float | Geographic longitude coordinate |
        | `latitude` | float | Geographic latitude coordinate |
        | `housing_median_age` | int | Median age of houses in the block |
        | `total_rooms` | int | Total number of rooms |
        | `total_bedrooms` | int | Total number of bedrooms |
        | `population` | int | Block population |
        | `households` | int | Number of households |
        | `median_income` | float | Median income (in 10,000s) |
        | `ocean_proximity` | string | Proximity to ocean |
        | `median_house_value` | float | Median house value (target) |
        """)

    with col3:
        st.markdown("""
        ### - Features:
        - **Comprehensive Statistics** - Detailed descriptive analytics
        - **Advanced Visualizations** - Interactive Plotly charts
        - **Geographic Analysis** - Map-based insights
        - **Feature Engineering** - Automatic feature creation
        - **Statistical Tests** - Hypothesis testing & validation
        - **Export Options** - Download processed datasets
        """)