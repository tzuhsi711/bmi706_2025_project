import altair as alt
import pandas as pd
import streamlit as st
import numpy as np
from vega_datasets import data

# Page configuration
st.set_page_config(
    layout="wide",
    page_title="Breast Cancer Clinical Trials"
)

# Disable Altair max rows limit
alt.data_transformers.disable_max_rows()


@st.cache_data
def breast_cancer_df():
    """
    Load and process the breast cancer clinical trials data.
    """
    # Load raw data
    breast_data = pd.read_csv('data/breast_cancer.csv', encoding='utf-8', encoding_errors='ignore')

    # Add country column (separate 'site_countries')
    breast_data = (
        breast_data
        .dropna(subset=['site_countries'])
        .assign(
            country=lambda c: c['site_countries'].str.split(r"\s*\|\s*")
        )
        .explode('country')
        .assign(country=lambda c: c['country'].str.strip())
    )

    # Investigate which countries only have one case per study status
    studyStatus_count = (
        breast_data
        .groupby(['country', 'study_status_grouped'])['nct_id']
        .nunique()
        .reset_index(name='n_trials')
    )

    # Identify countries with more than one case in all study statuses
    min_studyStatus_count = (
        studyStatus_count
        .groupby('country')['n_trials']
        .min()
    )
    countries_keep = min_studyStatus_count[min_studyStatus_count >= 2].index

    # Keep only countries in countries_keep
    breast_data = breast_data[breast_data['country'].isin(countries_keep)]

    # Convert to correct data type
    cat_vars = [
        'study_status_grouped',
        'source_class',
        'has_drug',
        'has_behavioral',
        'has_procedure',
        'has_device',
        'has_biological',
        'has_dmc_clean',
        'country'
    ]
    breast_data[cat_vars] = breast_data[cat_vars].astype('category')
    breast_data['start_year'] = pd.to_numeric(breast_data['start_year'], errors='coerce')

    # Remove nan, 2026, and 2027
    breast_data = breast_data[
        breast_data['start_year'].notna() &
        ~breast_data['start_year'].isin([2026, 2027])
    ]

    # Load coordinate data with proper encoding
    coord_data = pd.read_csv('data/world_coord.csv', encoding='utf-8', encoding_errors='ignore')
    coord_data = coord_data.drop(
        columns=['usa_state_code', 'usa_state_latitude', 'usa_state_longitude', 'usa_state']
    )

    # Add missing coordinate data
    missing_coord = pd.DataFrame({
        'country': [
            'Turkey (Türkiye)',
            'Czechia',
            'North Macedonia',
            'The Bahamas',
            "Côte d'Ivoire",
            'Republic of the Congo',
            'U.S. Minor Outlying Islands'
        ],
        'latitude': [
            38.9637,
            49.8175,
            41.6086,
            25.0343,
            7.5400,
            -0.66,
            19.2823,
        ],
        'longitude': [
            35.2433,
            15.4730,
            21.7453,
            -77.3963,
            -5.55,
            14.93,
            166.6470
        ]
    })
    missing_coord['country_code'] = np.nan

    # Add to coordinate dataframe
    coord_data = pd.concat([coord_data, missing_coord], ignore_index=True)

    # Merge coordinate data with breast cancer dataframe
    df = breast_data.merge(coord_data, on='country', how='left')

    return df


@st.cache_data
def intervention_df(df, selected_country="All", selected_status="All"):
    """
    Prepare intervention data in long format for visualization.
    """
    # Filter data 
    filtered_df = df.copy()
    if selected_country != "All":
        filtered_df = filtered_df[filtered_df['country'] == selected_country]
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df['study_status_grouped'] == selected_status]

    # Extract intervention columns
    intervention_cols = [col for col in filtered_df.columns if col.startswith('has')]
    intervention_cols.remove('has_dmc_clean')

    other_cols = [col for col in filtered_df.columns if col not in intervention_cols]

    # Melt to long format (NOW on filtered data)
    df_intervention_long = filtered_df.melt(
        id_vars=other_cols,
        value_vars=intervention_cols,
        var_name='intervention_type',
        value_name='has_intervention'
    )

    # Rename intervention types
    intervention_map = {
        'has_drug': 'Drug',
        'has_behavioral': 'Behavioral',
        'has_procedure': 'Procedure',
        'has_device': 'Device',
        'has_biological': 'Biological'
    }
    df_intervention_long['intervention_type'] = (
        df_intervention_long['intervention_type'].map(intervention_map)
    )

    # Remove rows where has_intervention == 0
    df_intervention_long = df_intervention_long[
        df_intervention_long['has_intervention'] == 1
    ]

    # Calculate max days based on year range
    max_days = (
        df_intervention_long['start_year'].max() -
        df_intervention_long['start_year'].min()
    ) * 365.25

    # Filter study duration
    df_intervention_long = df_intervention_long[
        (df_intervention_long['study_duration_days'].notna()) &
        (df_intervention_long['study_duration_days'] <= max_days)
    ]

    # Convert study duration to years
    df_intervention_long['study_duration_years'] = (
        df_intervention_long['study_duration_days'] / 365.25
    )

    # Define bins and labels for duration groups
    bins = [0, 1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, np.inf]
    labels = [
        "<1 year", "1-5 years", "6-10 years", "11-15 years",
        "16-20 years", "21-25 years", "26-30 years", "31-35 years",
        "36-40 years", "41-45 years", "46-50 years", "51-55 years", ">55 years"
    ]

    # Assign categories
    df_intervention_long['duration_year_group'] = pd.cut(
        df_intervention_long['study_duration_years'],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    )

    # Order categories
    df_intervention_long['duration_year_group'] = pd.Categorical(
        df_intervention_long['duration_year_group'],
        categories=labels,
        ordered=True
    )

    return intervention_df


@st.cache_data
def get_all_countries(df):
    """Get list of all countries in dataset."""
    return sorted(df['country'].unique())


@st.cache_data
def get_all_statuses(df):
    """Get list of all study statuses in dataset."""
    return sorted(df['study_status_grouped'].unique())


# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Load data
df = breast_cancer_df()

# Title and description
st.title("Breast Cancer Clinical Trials")
st.markdown(
    "Interactive analysis of breast cancer clinical trial trends, "
    "geographic distribution, and research characteristics"
)

# Filters in main content area
st.header("Filters")

# Initialize session state for filters if not exists
if 'filter_status' not in st.session_state:
    st.session_state.filter_status = "All"
if 'filter_country' not in st.session_state:
    st.session_state.filter_country = "All"

# Filters in horizontal layout
filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])

with filter_col1:
    # Study status selector - NO dependency on country
    all_statuses = get_all_statuses(df)
    st.selectbox(
        "Study Status",
        ["All"] + all_statuses,
        key="filter_status"
    )

with filter_col2:
    # Country selector - NO dependency on status
    all_countries = get_all_countries(df)
    st.selectbox(
        "Country",
        ["All"] + all_countries,
        key="filter_country"
    )

# Access filters from session state
selected_status = st.session_state.filter_status
selected_country = st.session_state.filter_country

# Prepare intervention data with current filters
df_intervention = intervention_data(df, selected_country, selected_status)

# Display result count feedback
if selected_country != "All" or selected_status != "All":
    result_count = len(df_intervention)
    if result_count == 0:
        st.warning(f"⚠️ No trials found for Status='{selected_status}' and Country='{selected_country}'. Try different filters.")
    else:
        st.info(f"Showing {result_count:,} trials with current filters")

with filter_col3:
    st.info(
        "**Year Selection**: Drag across the temporal trends chart to filter by year range. "
        "All charts respond to the filters and year range selection."
    )

st.markdown("---")


# Create Altair parameter bindings for filters
# Country dropdown menu
country_dropdown = alt.param(
    name='country',
    value=selected_country,
)

# Status dropdown menu
status_dropdown = alt.param(
    name='status',
    value=selected_status,
)

# Year brush selection
year_brush = alt.selection_interval(
    name='year',
    encodings=['x']
)


# ========================================================
# TASK 1: TEMPORAL TRENDS LINE CHART
# ========================================================

trial_lineChart = (
    alt.Chart(df, title='Temporal Trends in Breast Cancer Trials')
    .transform_filter(
        f"'{selected_country}' == 'All' || datum.country == '{selected_country}'"
    )
    .transform_filter(
        f"'{selected_status}' == 'All' || datum.study_status_grouped == '{selected_status}'"
    )
    .transform_aggregate(
        num_trial='count()',
        groupby=['start_year', 'study_status_grouped']
    )
    .mark_line(point=True)
    .encode(
        x=alt.X(
            'start_year:Q',
            title='Start Year',
            axis=alt.Axis(labelAngle=45, grid=False, format='d')
        ),
        y=alt.Y(
            'num_trial:Q',
            title='Number of Trials'
        ),
        color=alt.Color(
            'study_status_grouped:N',
            title='Study Status',
            scale=alt.Scale(scheme='category10')
        ),
        tooltip=[
            alt.Tooltip('start_year:Q', title='Year', format='d'),
            alt.Tooltip('study_status_grouped:N', title='Study Status'),
            alt.Tooltip('num_trial:Q', title='Trial Count')
        ]
    )
    .add_params(year_brush)
    .properties(width=500, height=200)
)


# ============================================================================
# TASK 2: GEOGRAPHIC DISTRIBUTION MAP 
# ============================================================================

# Load world map
world = alt.topo_feature(data.world_110m.url, 'countries')

# Background map
background = (
    alt.Chart(world)
    .mark_geoshape(fill='lightgray', stroke='white')
    .properties(width=500, height=400)
    .project('equirectangular')
)

# Points layer - Ohe circle per country
point = (
    alt.Chart(df, title='Geographic Distribution of Trials')
    .transform_filter(
        f"'{selected_country}' == 'All' || datum.country == '{selected_country}'"
    )
    .transform_filter(
        f"'{selected_status}' == 'All' || datum.study_status_grouped == '{selected_status}'"
    )
    .transform_filter(year_brush)
    .transform_aggregate(
        num_trial='count()',
        latitude='mean(latitude)',
        longitude='mean(longitude)',
        groupby=['country']
    )
    .mark_circle(opacity=0.6, color='steelblue')
    .encode(
        latitude='latitude:Q',
        longitude='longitude:Q',
        size=alt.Size(
            'num_trial:Q',
            title='Trial Count',
            scale=alt.Scale(range=[10, 1000])
        ),
        tooltip=[
            alt.Tooltip('country:N', title='Country'),
            alt.Tooltip('num_trial:Q', title='Trial Count')
        ]
    )
)

# Combine map layers
trial_mapChart = (background + point).properties(width=900, height=500)


# ============================================================================
# TASK 3: INTERVENTION TYPE VISUALIZATIONS
# ============================================================================

# Line chart - Temporal trends by intervention type
intervention_lineChart = (
    alt.Chart(df_intervention, title='Temporal Trends in Intervention Types')
    .transform_filter(
        f"'{selected_country}' == 'All' || datum.country == '{selected_country}'"
    )
    .transform_filter(
        f"'{selected_status}' == 'All' || datum.study_status_grouped == '{selected_status}'"
    )
    .transform_filter(year_brush)
    .transform_aggregate(
        num_trial='count()',
        groupby=['start_year', 'intervention_type']
    )
    .mark_line(point=True)
    .encode(
        x=alt.X(
            'start_year:Q',
            title='Year',
            axis=alt.Axis(labelAngle=45, grid=False, format='d')
        ),
        y=alt.Y(
            'num_trial:Q',
            title='Number of Trials'
        ),
        color=alt.Color(
            'intervention_type:N',
            title='Intervention Type',
            scale=alt.Scale(scheme='category10')
        ),
        tooltip=[
            alt.Tooltip('start_year:Q', title='Year', format='d'),
            alt.Tooltip('intervention_type:N', title='Intervention Type'),
            alt.Tooltip('num_trial:Q', title='Trial Count')
        ]
    )
    .properties(width=600, height=300)
)

# Stacked bar chart - Intervention composition by study duration
stackedBar_Chart = (
    alt.Chart(df_intervention, title='Intervention Type Composition vs Study Duration')
    .transform_filter(
        f"'{selected_country}' == 'All' || datum.country == '{selected_country}'"
    )
    .transform_filter(
        f"'{selected_status}' == 'All' || datum.study_status_grouped == '{selected_status}'"
    )
    .transform_filter(year_brush)
    .transform_aggregate(
        num_trial='count()',
        groupby=['duration_year_group', 'intervention_type']
    )
    .transform_joinaggregate(
        total_trial='sum(num_trial)',
        groupby=['duration_year_group']
    )
    .transform_calculate(
        prop='datum.num_trial / datum.total_trial'
    )
    .mark_bar()
    .encode(
        x=alt.X(
            'duration_year_group:O',
            axis=alt.Axis(labelAngle=45),
            title='Study Duration (Years)'
        ),
        y=alt.Y(
            'num_trial:Q',
            stack='normalize',
            title='Proportion of Trials'
        ),
        color=alt.Color(
            'intervention_type:N',
            title='Intervention Type',
            scale=alt.Scale(scheme='category10')
        ),
        tooltip=[
            alt.Tooltip('duration_year_group:O', title='Study Duration'),
            alt.Tooltip('intervention_type:N', title='Intervention Type'),
            alt.Tooltip('num_trial:Q', title='Trial Count'),
            alt.Tooltip('prop:Q', title='Proportion', format='.1%')
        ]
    )
    .properties(width=600, height=300)
)


# ============================================================================
# TASK 4: SPONSOR TYPE PIE CHART
# ============================================================================

pie_Chart = (
    alt.Chart(df, title='Sponsor Type Composition')
    .transform_filter(
        f"'{selected_country}' == 'All' || datum.country == '{selected_country}'"
    )
    .transform_filter(
        f"'{selected_status}' == 'All' || datum.study_status_grouped == '{selected_status}'"
    )
    .transform_filter('datum.source_class != null')
    .transform_filter(year_brush)
    .transform_aggregate(
        num_trial='count()',
        groupby=['source_class']
    )
    .transform_joinaggregate(
        total_trial='sum(num_trial)'
    )
    .transform_calculate(
        prop='datum.num_trial / datum.total_trial'
    )
    .mark_arc()
    .encode(
        theta=alt.Theta(
            'num_trial:Q',
            stack=True,
            title='Number of Trials'
        ),
        color=alt.Color(
            'source_class:N',
            title='Sponsor Type',
            scale=alt.Scale(scheme='category10')
        ),
        tooltip=[
            alt.Tooltip('source_class:N', title='Sponsor Type'),
            alt.Tooltip('num_trial:Q', title='Trial Count'),
            alt.Tooltip('prop:Q', title='Proportion', format='.1%')
        ]
    )
    .properties(width=400, height=300)
)


# ============================================================================
# FINAL DASHBOARD LAYOUT
# ============================================================================

# Combine all charts vertically to share the year_brush selection
combined_chart = alt.vconcat(
    trial_lineChart.properties(title='Temporal Trends by Study Status'),
    trial_mapChart.properties(title='Geographic Distribution'),
    intervention_lineChart.properties(title='Intervention Type - Temporal Trends'),
    stackedBar_Chart.properties(title='Intervention Type - Composition by Study Duration'),
    pie_Chart.properties(title='Sponsor Type Composition')
).resolve_scale(
    color='independent'
)

# Display the combined chart
st.altair_chart(combined_chart, use_container_width=True)

# Sponsor type explanation
with st.expander("Sponsor Category Definitions", expanded=False):
    st.markdown("""
    **INDUSTRY (20%)**: Pharmaceutical and biotech companies

    **OTHER (72%)**: Universities, academic medical centers,
    non-profit hospitals, and cancer research organizations

    **NIH (3.5%)**: US National Institutes of Health agencies

    **NETWORK (2%)**: Research consortia and cooperative groups

    **OTHER_GOV (2%)**: Non-US government hospitals and health agencies

    **FED (0.1%)**: US Federal government agencies excluding NIH

    **INDIV (<0.1%)**: Individual physician investigators
    """)

# Footer
st.markdown("---")
st.caption(
    "Data source: Breast Cancer Clinical Trials Database | "
    "Dashboard created with Streamlit and Altair"
)