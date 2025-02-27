import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go


class Warnings():

    @staticmethod
    def generate_no_ticker_pop(target_button):
        pop_estimation_no_ticker = dbc.Popover("Before estimating, please input the ticker in the search bar!",
                                               id="no-ticker-pop-"+target_button,
                                               target=target_button,
                                               is_open=False,
                                               className="w-25 p-3 bg-warning text-white",
                                               hide_arrow=True,
                                               )
        return pop_estimation_no_ticker

    @staticmethod
    def generate_no_date_pop(target_button,
                             content="Before estimating, please input the date in correct format!\n And make sure the date range falls within the available data range."):
        pop_no_date = dbc.Popover(content,
                                  id="no-date-pop-"+target_button,
                                  target=target_button,
                                  is_open=False,
                                  className="w-25 p-3 bg-warning text-white",
                                  hide_arrow=True,
                                  )
        return pop_no_date

    @staticmethod
    def generate_no_params_pop(target_button):
        pop_no_params = dbc.Popover("Please estimate the model first",
                                    id="no-params-pop-"+target_button,
                                    target=target_button,
                                    is_open=False,
                                    className="w-25 p-3 bg-warning text-white",
                                    hide_arrow=True,
                                    )
        return pop_no_params

    @staticmethod
    # generate_date_order_error("estimate-button")
    def generate_date_order_error(traget_button):
        pop_wrong_date_order = dbc.Popover("The estimated start date should be prior to the end date.",
                                           id="wrong-date-order-pop-"+traget_button,
                                           target=traget_button,
                                           is_open=False,
                                           className="w-25 p-3 bg-warning text-white",
                                           hide_arrow=True,
                                           )
        return pop_wrong_date_order

    @staticmethod
    def generate_no_params_pop(target_button):
        pop_no_params = dbc.Popover("Please estimate the model first",
                                    id="no-params-pop-"+target_button,
                                    target=target_button,
                                    is_open=False,
                                    className="w-25 p-3 bg-warning text-white",
                                    hide_arrow=True,
                                    )
        return pop_no_params

    @staticmethod
    def generate_no_int_pop(target_button):
        pop_no_int = dbc.Popover("The input should be an integer",
                                 id="no-int-pop-"+target_button,
                                 target=target_button,
                                 is_open=False,
                                 className="w-25 p-3 bg-warning text-white",
                                 hide_arrow=True,
                                 )
        return pop_no_int

    @staticmethod
    def generate_lack_action_pop(target_button,
                                 content='Before doing sensitivity analysis, please do the monte carlo simulation first!'):
        pop = dbc.Popover(content,
                          id="lack-action-pop-"+target_button,
                          target=target_button,
                          is_open=False,
                          className="w-25 p-3 bg-warning text-white",
                          hide_arrow=True,
                          )
        return pop

    @staticmethod
    def generate_short_window_pop(target_button,
                                  content='The time window is too short. Please input longer one.'):
        pop = dbc.Popover(content,
                          id="short-window-pop-"+target_button,
                          target=target_button,
                          is_open=False,
                          className="w-25 p-3 bg-warning text-white",
                          hide_arrow=True,
                          )
        return pop


class Graphs():

    @staticmethod
    def get_empty_ts_fig():

        fig = go.Figure()

        fig.update_layout(
            title={
                'y': 0.95,
                'x': 0.5,
                'font': {'size': 18}
            },
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis={
                'title': 'Closing Date',
                'showline': True,
                'linewidth': 1,
                'linecolor': 'black'
            },
            yaxis={
                'showline': True,
                'linewidth': 1,
                'linecolor': 'black'
            }
        )

        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="todate"
                             ),
                        dict(count=6,
                             label="6m",
                             step="month",
                             # stepmode="backward"
                             ),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             # stepmode="todate"
                             ),
                        dict(count=2,
                             label="1y",
                             step="year",
                             # stepmode="backward"
                             ),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date",
            )
        )

        return (fig)
