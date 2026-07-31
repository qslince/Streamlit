import streamlit as st
import polars as pl
import requests
from datetime import date
from utils import (SEASON_ORDER, get_season, make_trends_graph,
                   make_anomal_timeseries_graph, make_season_profile_graph)


@st.cache_data
def load_data(file):
    return pl.read_csv(file, try_parse_dates=True)

st.title("Analysis of temperature changes in cities over time")

st.header("Dataset uploading")

uploaded_file = st.file_uploader('temperature_data.csv', type=['csv'])

if uploaded_file is None:
    st.write('Please, upload .csv file')

else:
    data = load_data(uploaded_file)

    city = st.selectbox('Choose city for analysis', options=data['city'].unique(maintain_order=True).to_list())

    data = (
        data.with_columns(
            avg_city_season_temp = pl.mean('temperature').over(["city", "season"]),
            std_city_season_temp = pl.std('temperature').over(["city", "season"])
        )
    )

    city_df = data.filter(pl.col('city') == city)
    st.dataframe(city_df.head(10))

    st.header("Today`s weather")
    weather_api = st.text_input('Enter API-key for OpenWeatherMap:', type='password')

    if st.button("Get weather forecast"):
        if not weather_api:
            st.warning('Please, enter API-key')
            st.stop()
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api}&units=metric"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data_weather = response.json()
            st.write(f"Today temperature in {city} is {data_weather['main']['temp']}°C. It feels like {data_weather['main']['feels_like']}°C!")

            today = date.today()
            cur_season = get_season(today)

            avg_temp, std_temp = city_df.filter(pl.col('season') == cur_season).select(pl.col('avg_city_season_temp'), pl.col('std_city_season_temp')).row(0)

            if data_weather['main']['temp'] < avg_temp - 2* std_temp or data_weather['main']['temp'] > avg_temp + 2 * std_temp:
                st.write("Today`s temperature is anomal")

            else:
                st.write("Today`s temperature is normal")

        except requests.exceptions.Timeout:
            st.error("Request timed out")

        except requests.exceptions.ConnectionError:
            st.error("Error connecting to the server. Check your internet connection.")

        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                st.error("Invalid API key. Please see https://openweathermap.org/faq#error401 for more info.")

            elif response.status_code == 404:
                st.error(f"City {city} was not found")

            else:
                st.error(f"Error HTTP: {e}")

    st.header(f"Historical information on {city}")

    hist_stats = (
        city_df.group_by('season')
        .agg(average_temp=pl.mean('temperature'), std_temp=pl.std('temperature'),
             min_temp=pl.min('temperature'), max_temp=pl.max('temperature'),
             median_temp=pl.median('temperature'))
        .with_columns(o=pl.col('season').replace_strict(
            {s: i for i, s in enumerate(SEASON_ORDER)}, return_dtype=pl.Int8))
        .sort('o').drop('o')
        .with_columns(pl.selectors.float().round(2))
    )

    st.dataframe(hist_stats, use_container_width=True, hide_index=True)

    trends_graph = make_trends_graph(city_df)
    st.plotly_chart(trends_graph, use_container_width=True)


    city_df = (
        city_df.with_columns(
            is_anomal_temp= ((pl.col('temperature') > pl.col('avg_city_season_temp') + 2 * pl.col('std_city_season_temp')) 
            |  (pl.col('temperature') < pl.col('avg_city_season_temp') - 2 * pl.col('std_city_season_temp'))
            )
        )
    )

    anoms_graph = make_anomal_timeseries_graph(city_df)
    st.plotly_chart(anoms_graph,use_container_width=True)

    st.header("Season profile")
    
    profile_fig, profile_df = make_season_profile_graph(city_df)
    left, right = st.columns([2, 1])

    left.plotly_chart(profile_fig, use_container_width=True)
    right.dataframe(profile_df.with_columns(pl.selectors.float().round(2)),
                    use_container_width=True, hide_index=True)
    
    last_year = city_df['timestamp'].max().year

    stats_all = (
        city_df.group_by('season')
        .agg(average_temp=pl.mean('temperature'), std_temp=pl.std('temperature'))
    )

    stats_last = (
        city_df.filter(pl.col('timestamp').dt.year() == last_year)
        .group_by('season')
        .agg(average_temp_last=pl.mean('temperature'), std_temp_last=pl.std('temperature'))
    )

    city_season_stats = (
        stats_all.join(stats_last, on='season')
        .with_columns(o=pl.col('season').replace_strict(
            {s: i for i, s in enumerate(SEASON_ORDER)}, return_dtype=pl.Int8))
        .sort('o').drop('o')
        .with_columns(diff=pl.col('average_temp_last') - pl.col('average_temp'))
        .with_columns(pl.selectors.float().round(2))
    )

    st.caption(f"Seasonal averages over the whole period vs {last_year}")
    st.dataframe(city_season_stats, use_container_width=True, hide_index=True)
