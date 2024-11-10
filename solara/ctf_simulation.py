import numpy as np
import plotly.express as px
import solara


@solara.component
def Page():
    solara.Title("CTF Simulation")
    google_analytics(id="G-YV3ZFR8VG6")
    
    defocus, set_defocus = solara.use_state(0.5)
    voltage, set_voltage = solara.use_state(300)
    cs, set_cs = solara.use_state(2.7)
    ampcontrast, set_ampcontrast = solara.use_state(7)
    phaseshift, set_phaseshift = solara.use_state(0)
    apix, set_apix = solara.use_state(1.0)
    imagesize, set_imagesize = solara.use_state(256)
    over_sample, set_over_sample = solara.use_state(1)

    with solara.Sidebar():
        solara.InputFloat("Defocus (μm)", value=defocus, on_value=set_defocus)
        solara.SliderFloat("", value=defocus, min=0, max=5, step=0.1, on_value=set_defocus)
        solara.InputFloat("Voltage (kV)", value=voltage, on_value=set_voltage)
        solara.InputFloat("Spherical aberration (mm)", value=cs, on_value=set_cs)
        solara.InputFloat("Amplitude contrast (%)", value=ampcontrast, on_value=set_ampcontrast)
        solara.InputFloat("Phase shift (°)", value=phaseshift, on_value=set_phaseshift)
        solara.InputFloat("Pixel size (Å)", value=apix, on_value=set_apix)
        solara.InputFloat("Image size (pixels)", value=imagesize, on_value=set_imagesize)
        solara.InputFloat("Oversampling", value=over_sample, on_value=set_over_sample)
    
    ctf = CTF(
        defocus=defocus,
        voltage=voltage,
        cs=cs,
        ampcontrast=ampcontrast,
        phaseshift=phaseshift,
        apix=apix,
        imagesize=imagesize,
        over_sample=over_sample,
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

    solara.FigurePlotly(fig)

    solara.Markdown(
        '*Developed by the [Jiang Lab](https://jiang.bio.purdue.edu). Report issues at [CTFSimulation@GitHub](https://github.com/jianglab/ctfsimulation/issues)*'
    )

def google_analytics(id):
    solara.HTML(
        tag="script",
        attributes={"src": f"https://www.googletagmanager.com/gtag/js?id={id}"}
    )
    solara.HTML(
        tag="script",
        unsafe_innerHTML=f"""window.dataLayer = window.dataLayer || []; function gtag(){{dataLayer.push(arguments);}}
                gtag('js', new Date()); gtag('config', '{id}');"""
    )

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

