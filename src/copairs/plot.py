from typing import Optional
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from copairs.replicating import CorrelationTestResult


def plot(
    corr_score: CorrelationTestResult,
    percent_score: float,
    title: str,
    left_null_th: Optional[float] = None,
    right_null_th: Optional[float] = None,
    true_dist_title="True replicates",
    null_dist_title="Null distribution",
) -> go.Figure:
    """
    Plot two distributions and threshold(s) line.
    """
    # fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Histogram(
            x=corr_score.corr_dist,
            nbinsx=20,
            # histnorm='probability',
            name=true_dist_title,
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Histogram(
            x=corr_score.null_dist,
            nbinsx=100,
            # histnorm='probability',
            name=null_dist_title,
        )
    )
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(title_text=true_dist_title, secondary_y=True)
    fig.update_yaxes(title_text=null_dist_title, secondary_y=False)

    for pos, null_th in [("left", left_null_th), ("right", right_null_th)]:
        if null_th:
            fig.add_vline(
                x=null_th,
                line_width=3,
                line_dash="dash",
                line_color="black",
                annotation_text=f" Null th:{null_th:0.2}",
                annotation_position=f"top {pos}",
            )

    fig.update_traces(opacity=0.75, marker_line_width=1)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )
    fig.update_layout(
        font=dict(
            size=22,
        )
    )

    fig.add_annotation(
        text=f"{title}: {percent_score:0.1%}",
        x=0,
        y=1.06,
        showarrow=False,
        bgcolor="#ffffff",
        xref="paper",
        yref="paper",
        yanchor="bottom",
        xanchor="left",
        font=dict(size=20),
    )

    return fig
