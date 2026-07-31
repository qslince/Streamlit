import scipy.stats as stats
import plotly.graph_objects as go
import polars as pl

SEASON_ORDER = ['winter', 'spring', 'summer', 'autumn']

def get_season(date):
    if date.month in [12, 1, 2]:
        return "winter"
    elif date.month in [3, 4, 5]:
        return "spring"
    elif date.month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"



def make_trends_graph(city_df):

    city_df = city_df.with_columns(
        non_seasonal_temperature = pl.col('temperature') - pl.mean('temperature').over(['city', 'season'])
    )
    group = city_df.sort('timestamp')
    
    days = group['timestamp'].dt.epoch('d')
    days = (days - days[0]) / 365.
        
    reg = stats.linregress(days.to_numpy(), group['non_seasonal_temperature'].to_numpy())
    
    slope, intercept = reg.slope, reg.intercept

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=group['timestamp'],
        y=group['non_seasonal_temperature'],
        mode='lines',
        name='deseasonalized temperature',
        line=dict(width=0.6, color='rgba(120,120,120,.5)'),
    ))
    fig.add_trace(go.Scatter(
        x=[group['timestamp'][0], group['timestamp'][-1]],
        y=[intercept, intercept + slope * days[-1]],
        mode='lines',
        name=f'trend {slope * 10:+.2f} °C/decade (p={reg.pvalue:.3f})',
        line=dict(width=3, color='red'),
    ))
    fig.add_hline(y=0, line=dict(width=1, dash='dot', color='#888'))

    fig.update_layout(
        title=f"Long-term trend, {group['timestamp'][0].year}–{group['timestamp'][-1].year}",
        xaxis_title="Date",
        yaxis_title="°C from seasonal norm",
        hovermode='x unified',
        height=420,
    )

    return fig

def make_anomal_timeseries_graph(city_df):
    city_df = city_df.sort('timestamp')
    anom = city_df.filter(pl.col('is_anomal_temp'))
    ts = city_df['timestamp'].to_list()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ts, y=city_df['avg_city_season_temp'] + 2 * city_df['std_city_season_temp'],
        line=dict(width=0), showlegend=False, hoverinfo='skip'))
    
    fig.add_trace(go.Scatter(
        x=ts, y=city_df['avg_city_season_temp'] - 2 * city_df['std_city_season_temp'],
        line=dict(width=0), fill='tonexty', fillcolor='rgba(99,110,250,.13)',
        name='seasonal mean ± 2σ', hoverinfo='skip'))

    fig.add_trace(go.Scatter(
        x=ts, y=city_df['temperature'], mode='lines', name='daily temperature',
        line=dict(width=0.6, color='rgba(120,120,120,.5)')))

    fig.add_trace(go.Scatter(
        x=ts, y=city_df['temperature'].rolling_mean(30), mode='lines',
        name='30-day moving average', line=dict(width=2, color='blue')))

    fig.add_trace(go.Scatter(
        x=anom['timestamp'].to_list(), y=anom['temperature'],
        mode='markers', name=f'anomalies ({len(anom)})',
        marker=dict(size=5, color='red')))

    fig.update_layout(title='Daily temperature, moving average and anomalies',
                      xaxis_title='Date', yaxis_title='°C',
                      hovermode='x unified', height=460)
    return fig

def make_season_profile_graph(city_df):
    prof = (city_df.group_by('season')
            .agg(mean=pl.mean('temperature'), std=pl.std('temperature'))
            .with_columns(o=pl.col('season').replace_strict(
                {s: i for i, s in enumerate(SEASON_ORDER)}, return_dtype=pl.Int8))
            .sort('o').drop('o'))

    fig = go.Figure(go.Bar(
        x=prof['season'], y=prof['mean'],
        error_y=dict(type='data', array=(2 * prof['std']).to_list(),
                     color='#444', thickness=1.5),
        marker_color=['blue', 'green', 'red', 'orange'],
        hovertemplate='%{x}<br>mean %{y:.1f} °C<extra></extra>'))

    fig.update_layout(title='Seasonal profile (whiskers = ±2σ)',
                      xaxis_title='Season', yaxis_title='°C', height=380)
    return fig, prof