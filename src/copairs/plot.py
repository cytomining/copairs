from plotly import graph_objects as go
from plotly.subplots import make_subplots

from replicating import CorrelationTestResult


def plot(corr_score: CorrelationTestResult, percent_score: float,
         null_th: float, title: str) -> go.Figure:
    '''
        Plot two distributions and a threshold line.
        '''
    # fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Histogram(
            x=corr_score.corr_dist,
            nbinsx=20,
            # histnorm='probability',
            name='True replicates'),
        secondary_y=True)
    fig.add_trace(
        go.Histogram(
            x=corr_score.null_dist,
            nbinsx=100,
            # histnorm='probability',
            name='Null distribution'))
    fig.update_layout(barmode='overlay')
    fig.update_yaxes(title_text='True replicates', secondary_y=True)
    fig.update_yaxes(title_text='Null distribution', secondary_y=False)

    fig.add_vline(x=null_th,
                  line_width=3,
                  line_dash="dash",
                  line_color="black",
                  annotation_text=f' Null threshold:{null_th:0.2}',
                  annotation_position="top right")

    fig.update_traces(opacity=0.75, marker_line_width=1)
    fig.update_layout(legend=dict(
        orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1))
    fig.update_layout(font=dict(size=22, ))

    fig.add_annotation(text=f'{title}: {percent_score:0.1%}',
                       x=0,
                       y=1.06,
                       showarrow=False,
                       bgcolor="#ffffff",
                       xref='paper',
                       yref='paper',
                       yanchor='bottom',
                       xanchor='left',
                       font=dict(size=20))

    return fig
