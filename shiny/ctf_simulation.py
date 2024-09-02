from shiny import reactive
from shiny.express import input, ui, render
from shinywidgets import render_plotly
import plotly.express as px
import numpy as np


class CTF:
    def __init__(
        self,
        voltage=300.0,
        cs=2.7,
        ampcontrast=7.0,
        defocus=0.5,
        dfdiff=0.0,
        dfang=0.0,
        phaseshift=0.0,
        bfactor=0.0,
        alpha=0.0,
        cc=2.7,
        dE=0.0,
        dI=0.0,
        dZ=0.0,
        dXY=0.0,
        apix=1.0,
        imagesize=256,
        over_sample=1,
        ctf_type="CTF",
    ):
        self.voltage = voltage
        self.cs = cs
        self.ampcontrast = ampcontrast
        self.defocus = defocus
        self.dfdiff = dfdiff
        self.dfang = dfang
        self.phaseshift = phaseshift
        self.bfactor = bfactor
        self.alpha = alpha
        self.cc = cc
        self.dE = dE
        self.dI = dI
        self.dZ = dZ
        self.dXY = dXY
        self.apix = apix
        self.imagesize = int(imagesize)
        self.over_sample = int(over_sample)
        self.ctf_type = ctf_type

    def wave_length(self):
        wl = 12.2639 / np.sqrt(
            self.voltage * 1000.0 + 0.97845 * self.voltage * self.voltage
        )  # Angstrom
        return wl

    def ctf1d(self):
        s_nyquist = 1.0 / (2 * self.apix)
        ds = s_nyquist / (self.imagesize // 2 * self.over_sample)
        s = np.arange(self.imagesize // 2 * self.over_sample + 1, dtype=np.float32) * ds
        s2 = s * s
        wl = self.wave_length()
        phaseshift = self.phaseshift * np.pi / 180.0 + np.arcsin(
            self.ampcontrast / 100.0
        )
        gamma = (
            2
            * np.pi
            * (
                -0.5 * self.defocus * 1e4 * wl * s2
                + 0.25 * self.cs * 1e7 * wl**3 * s2**2
            )
            - phaseshift
        )
        ctf = np.sin(gamma)
        return s, ctf


# ui.page_opts(title="Penguins dashboard", fillable=True)

with ui.sidebar():
    ui.input_numeric("defocus", "Defocus (μm)", value=0.5, min=0, step=0.1)
    ui.input_slider("defocus_slider", "", min=0, max=5, value=0.5, step=0.1)
    ui.input_numeric("voltage", "Voltage (kV)", value=300, min=100, step=100)
    ui.input_numeric("cs", "Spherical aberration (mm)", value=2.7, min=0, step=0.1)
    ui.input_numeric(
        "ampcontrast", "Amplitude contrast (%)", value=7, min=0, max=100, step=1
    )
    ui.input_numeric("phaseshift", "Phase shift (°)", value=0, min=0, max=360, step=10)
    ui.input_numeric("apix", "Pixel size (Å)", value=1.0, min=0.1, step=0.1)
    ui.input_numeric("imagesize", "Image size (pixels)", value=256, min=64, step=64)
    ui.input_numeric("over_sample", "Oversampling", value=1, min=1, step=1)

    @reactive.Effect
    @reactive.event(input.defocus)
    def _():
        ui.update_slider(
            "defocus_slider",
            value=input.defocus(),
            min=input.defocus() / 2,
            max=max(0.1, input.defocus() * 2),
            step=max(round(input.defocus() / 20, 6), 0.0001),
        )

    @reactive.Effect
    @reactive.event(input.defocus_slider)
    def _():
        ui.update_numeric("defocus", value=input.defocus_slider())


ui.h1("CTF Simulation", align="center")

with ui.card(full_screen=True):

    @render_plotly
    def plot():
        ctf = CTF(
            defocus=input.defocus(),
            voltage=input.voltage(),
            cs=input.cs(),
            ampcontrast=input.ampcontrast(),
            phaseshift=input.phaseshift(),
            apix=input.apix(),
            imagesize=input.imagesize(),
            over_sample=input.over_sample(),
        )
        s, ctf_values = ctf.ctf1d()
        fig = px.line(
            x=s,
            y=ctf_values,
            labels={"x": "Spatial Frequency (1/Å)", "y": "CTF"},
            height=650,
            markers=True,
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                showgrid=True,
                gridcolor="black",
                gridwidth=0.5,
                zeroline=False,
                linecolor="black",
                griddash="dash",
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="black",
                gridwidth=0.5,
                zeroline=False,
                linecolor="black",
                griddash="dash",
            ),
        )
        fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
        return fig


@render.ui
def _():
    return ui.div(
        ui.tags.i(
            "Developed by the ",
            ui.tags.a(
                "Jiang Lab", href="https://jiang.bio.purdue.edu", target="_blank"
            ),
            ". Report issues at ",
            ui.tags.a(
                "CTFSimulation@GitHub",
                href="https://github.com/jianglab/ctfsimulation/issues",
                target="_blank",
            ),
        )
    )
