""" 
MIT License

Copyright (c) 2020-2021 Wen Jiang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

def import_with_auto_install(packages, scope=locals()):
    if isinstance(packages, str): packages=[packages]
    for package in packages:
        if package.find(":")!=-1:
            package_import_name, package_pip_name = package.split(":")
        else:
            package_import_name, package_pip_name = package, package
        try:
            scope[package_import_name] = __import__(package_import_name)
        except ImportError:
            import subprocess
            subprocess.call(f'pip install {package_pip_name}', shell=True)
            scope[package_import_name] =  __import__(package_import_name)
required_packages = "streamlit numpy scipy bokeh skimage:scikit_image".split()
import_with_auto_install(required_packages)

import streamlit as st
import numpy as np

def main():
    session_state = SessionState_get(ctfs=[CTF()], emd_id=0)
    query_params = st.experimental_get_query_params()
    embed = "embed" in query_params

    title = "CTF Simulation"
    st.set_page_config(page_title=title, layout="wide")

    if embed:
        col1, col2 = st.beta_columns((1, 5))
    else:
        st.title(title)
        col1 = st.sidebar
        col2, col3 = st.beta_columns((5, 2))

    with col1:    # sidebar at the left of the screen
        ctfs = session_state.ctfs
        if embed:
            ctf_type = 'CTF'
            plot_abs = 0
            n = 1
        else:
            options = ('CTF', '|CTF|', 'CTF^2')
            ctf_type = st.selectbox(label='CTF type', options=options)
            plot_abs = options.index(ctf_type)
            n = st.number_input('# of CTFs', value=1, min_value=1, step=1)
        if n>len(ctfs):
            ctfs += [ CTF() for i in range(n-len(ctfs)) ]
        if n>1:
            i = st.number_input('CTF i=?', value=1, min_value=1, max_value=n, step=1)
            i -= 1
        else:
            i = 0
        value = ctfs[i].defocus if n>1 else 0.5
        ctfs[i].defocus = st.number_input('defocus (µm)', value=value, min_value=0.0, step=0.1, format="%.5g", key=f"defocus-{i}")
        if embed:
            rotavg = False
        else:
            value = ctfs[i].dfdiff if n>1 else 0.0
            ctfs[i].dfdiff = st.number_input('astigmatism mag (µm)', value=value, min_value=0.0, step=0.01, format="%g")
            if n==1 and ctfs[i].dfdiff:
                value = ctf_type=='CTF^2'
                rotavg = st.checkbox(label='plot rotational average', value=value)
            else:
                rotavg = False
            value = ctfs[i].dfang if n>1 else 0.0
            ctfs[i].dfang = st.number_input('astigmatism angle (°)', value=value, min_value=0.0, max_value=360., step=1.0, format="%g")
        value = ctfs[i].phaseshift if n>1 else 0.0
        ctfs[i].phaseshift = st.number_input('phase shift (°)', value=value, min_value=0.0, max_value=360., step=1.0, format="%g")
        apix = st.number_input('pixel size (Å/pixel)', value=1.0, min_value=0.1, step=0.01, format="%g")
        if embed:
            imagesize = 2048
            over_sample = 1
        else:
            imagesize = st.number_input('image size (pixel)', value=256, min_value=32, max_value=4096, step=4)
            over_sample = st.slider('over-sample (1x, 2x, 3x, etc)', value=1, min_value=1, max_value=6, step=1)
        
            with st.beta_expander("envelope functions", expanded=False):
                value = ctfs[i].bfactor if n>1 else 0.0
                ctfs[i].bfactor = st.number_input('b-factor (Å^2)', value=value, min_value=0.0, step=10.0, format="%g")
                value = ctfs[i].alpha if n>1 else 0.0
                ctfs[i].alpha = st.number_input('beam convergence semi-angle (mrad)', value=value, min_value=0.0, step=0.05, format="%g")
                value = ctfs[i].dE if n>1 else 0.0
                ctfs[i].dE = st.number_input('energy spread (eV)', value=value, min_value=0.0, step=0.2, format="%g")
                value = ctfs[i].dI if n>1 else 0.0
                ctfs[i].dI = st.number_input('objective lens current spread (ppm)', value=value, min_value=0.0, step=0.2, format="%g")
                if ctfs[i].dE or ctfs[i].dI:
                    value = ctfs[i].cc if n>1 else 2.7
                    ctfs[i].cc = st.number_input('cc (mm)', value=value, min_value=0.0, step=0.1, format="%g")
                value = ctfs[i].dZ if n>1 else 0.0
                ctfs[i].dZ = st.number_input('sample vertical motion (Å)', value=value, min_value=0.0, step=20.0, format="%g")
                value = ctfs[i].dXY if n>1 else 0.0
                ctfs[i].dXY = st.number_input('sample horizontal motion (Å)', value=value, min_value=0.0, step=0.2, format="%g")

        value = ctfs[i].voltage if n>1 else 300.0
        ctfs[i].voltage = st.number_input('voltage (kV)', value=value, min_value=10., step=100., format="%g")
        value = ctfs[i].cs if n>1 else 2.7
        ctfs[i].cs = st.number_input('cs (mm)', value=value, min_value=0.0, step=0.1, format="%g")
        value = ctfs[i].ampcontrast if n>1 else 7.0
        ctfs[i].ampcontrast = st.number_input('amplitude contrast (percent)', value=value, min_value=0.0, max_value=100., step=10.0, format="%g")

    with col2:
        if not embed:
            plot1d_s2 = st.checkbox(label='plot s^2 as x-axis', value=False)
        else:
            plot1d_s2 = False

        from bokeh.plotting import figure
        from bokeh.models import LegendItem
        if plot1d_s2:
            x_label = "s^2 (1/Å^2)"
            hover_x_var = "s^2"
            hover_x_val = "$x 1/Å^2"
        else:
            x_label = "s (1/Å)"
            hover_x_var = "s"
            hover_x_val = "$x 1/Å"
        y_label = f"{ctf_type}"

        tools = 'box_zoom,crosshair,hover,pan,reset,save,wheel_zoom'
        hover_tips = [("Res", "@res Å"), (hover_x_var, hover_x_val), (f"{ctf_type}", "$y")]
        if n>1 or (n==1 and ctfs[0].dfdiff):
            hover_tips = [("Defocus", "@defocus µm")] + hover_tips
        fig = figure(title="", x_axis_label=x_label, y_axis_label=y_label, tools=tools, tooltips=hover_tips)
        fig.title.align = "center"
        fig.title.text_font_size = "18px"

        from bokeh.palettes import Category10
        colors = Category10[10]
        line_dashes = 'dashed solid dotted dotdash dashdot'.split()

        legends = []
        raw_data = []
        for i in range(n):
            color = colors[ i % len(colors) ]
            if n==1 and ctfs[i].dfdiff:
                defocuses = [ctfs[i].defocus - ctfs[i].dfdiff, ctfs[i].defocus, ctfs[i].defocus + ctfs[i].dfdiff]
            else:
                defocuses = [ctfs[i].defocus]
            for di, defocus in enumerate(defocuses):
                s, s2, ctf = ctfs[i].ctf1d(apix, imagesize, over_sample, plot_abs, plot1d_s2, defocus_override=defocus)
                x = s2 if plot1d_s2 else s
                source = dict(x=x, res=1/s, y=ctf)
                if n>1 or (n==1 and ctfs[0].dfdiff): source["defocus"] = [defocus] * len(x)
                line_dash = line_dashes[di] if len(defocuses)>1 else "solid"
                line_width = 2 if len(defocuses)==1 or di==1 else 1
                line = fig.line(x='x', y='y', color=color, source=source, line_dash=line_dash, line_width=line_width)
                label = f"{round(defocus, 4):g} µm"
                legends.append(LegendItem(label=label, renderers=[line]))
                raw_data.append((label, ctf))

            if n==1 and rotavg:
                _, _, ctf_2d = ctfs[i].ctf2d(apix, imagesize, over_sample, plot_abs, plot1d_s2)
                rad_profile = compute_radial_profile(ctf_2d)
                source = dict(x=s2 if plot1d_s2 else s, res=1/s, y=rad_profile)
                line = fig.line(x='x', y='y', source=source, color='red', line_dash="solid", line_width=2)
                label = "rotavg"
                legends.append(LegendItem(label="rotavg", renderers=[line]))
                raw_data.append((label, rad_profile))

        fig.x_range.start = 0
        fig.x_range.end = source['x'][-1]
        fig.y_range.start = -1 if ctf_type == 'CTF' else 0
        fig.y_range.end = 1
        if len(legends)>1:
            from bokeh.models import Legend
            legend = Legend(items=legends, location="top_center", spacing=10, orientation="horizontal")
            fig.add_layout(legend, "above")
            fig.legend.click_policy= "hide"
            from bokeh.models import CustomJS
            from bokeh.events import MouseMove, DoubleTap
            toggle_legend_js = CustomJS(args=dict(leg=fig.legend[0]), code="""
                if (leg.visible) {
                    leg.visible = false
                    }
                else {
                    leg.visible = true
                }
            """)
            fig.js_on_event(DoubleTap, toggle_legend_js)
        st.bokeh_chart(fig, use_container_width=True)

        if not embed:
            show_data = st.checkbox('show raw data', value=False)
            if show_data:
                import pandas as pd
                data = np.zeros((len(x), 2 + len(raw_data)))
                data[:,0] = x
                data[:,1] = 1./s
                columns = [x_label, "Res (Å)"]
                for i, (label, ctf) in enumerate(raw_data):
                    data[:,2+i] = ctf
                    if len(raw_data)>1:
                        columns.append(y_label+' '+label)
                    else:
                        columns.append(y_label)
                columns = [col.rjust(12) for col in columns]
                df = pd.DataFrame(data, columns=columns)
                st.dataframe(df, width=None)
                st.markdown(get_table_download_link(df), unsafe_allow_html=True)

            st.markdown("**Learn more about [Contrast Transfer Function (CTF)](https://en.wikipedia.org/wiki/Contrast_transfer_function):**\n* [Getting Started in CryoEM, Grant Jensen](https://www.youtube.com/watch?v=mPynoF2j6zc&t=2s)\n* [CryoEM Principles, Fred Sigworth](https://www.youtube.com/watch?v=Y8wivQTJEHQ&list=PLRqNpJmSRfar_z87-oa5W421_HP1ScB25&index=5)\n")

        st.markdown("*Developed by the [Jiang Lab@Purdue University](https://jiang.bio.purdue.edu). Report problems to Wen Jiang (jiang12 at purdue.edu)*")

    if embed: return

    with col3:
        plot2d_s2 = st.checkbox(label='plot s^2 as radius', value=False)
        st.text("") # workaround for a layout bug in streamlit 

        show_color = False

        fig2ds = []
        for i in range(n):
            ds, ds2, ctf_2d = ctfs[i].ctf2d(apix, imagesize, over_sample, plot_abs, plot2d_s2)
            dxy = ds2 if plot2d_s2 else ds
            if n>1:
                title = f"{ctf_type} - {i+1}"
            else:
                title = f"{ctf_type}"
            fig2d = generate_image_figure(ctf_2d, dxy, ctf_type, title, plot2d_s2, show_color)
            fig2ds.append(fig2d)
        if len(fig2ds)>1:
            from bokeh.models import CrosshairTool
            crosshair = CrosshairTool(dimensions="both")
            crosshair.line_color = 'red'
            for fig in fig2ds: fig.add_tools(crosshair)
            from bokeh.layouts import gridplot
            figs_grid = gridplot(children=[fig2ds], toolbar_location=None)
            st.bokeh_chart(figs_grid, use_container_width=True)
        else:
            st.bokeh_chart(fig2d, use_container_width=True)
        
        with st.beta_expander("Simulate the CTF effect"):
            input_modes = ["Delta Function"]
            emdb_ids = get_emdb_ids()
            if emdb_ids:
                input_modes += ["Random EMDB ID", "Input an EMDB ID"]
            input_modes += ["Input an image url"]
            input_mode = st.radio(label="Choose an input mode:", options=input_modes, index=0)
            if input_mode == "Random EMDB ID":
                st.button(label="Change EMDB ID")
                import random
                emd_id = random.choice(emdb_ids)
                input_txt = f"EMD-{emd_id}"
            elif input_mode == "Input an EMDB ID":
                if session_state.emd_id==0:
                    import random
                    emd_id = random.choice(emdb_ids)
                else:
                    emd_id = session_state.emd_id        
                label = "Input an EMDB ID"
                value = f"EMD-{emd_id}"
                input_txt = st.text_input(label=label, value=value).strip()
            elif input_mode == "Input an image url":
                label = "Input an image url:"
                value = "https://images-na.ssl-images-amazon.com/images/I/61pSCxXEP8L._AC_SL1000_.jpg"
                input_txt = st.text_input(label=label, value=value).strip()
        
        image = None
        link = None
        if input_mode == "Delta Function":
            image = np.zeros_like(ctf_2d)
            ny, nx = image.shape
            image[ny//2, nx//2] = 255
        elif emdb_ids and input_txt.startswith("EMD-"):
            emd_id = input_txt[4:]
            if emd_id in emdb_ids:
                session_state.emd_id = emd_id
                image = get_emdb_image(emd_id, invert_contrast=-1, rgb2gray=True, output_shape=(imagesize*over_sample, imagesize*over_sample))
                #link = f'[EMD-{emd_id}](https://www.ebi.ac.uk/pdbe/entry/emdb/EMD-{emd_id})'
                link = f'[EMD-{emd_id}](https://www.emdataresource.org/EMD-{emd_id})'
            else:
                emd_id_bad = emd_id
                emd_id = random.choice(emdb_ids)
                st.warning(f"EMD-{emd_id_bad} does not exist. Please input a valid id (for example, a randomly selected valid id {emd_id})")
        elif input_txt.startswith("http") or input_txt.startswith("ftp"):   # input is a url
            url = input_txt
            image = get_image(url, invert_contrast=0, rgb2gray=True, output_shape=(imagesize*over_sample, imagesize*over_sample))
            if image is not None:
                image = image[::-1, :]
                link = f'[Image Link]({url})'
            else:
                st.warning(f"{url} is not a valid image link")
        elif len(input_txt):
            st.warning(f"{input_txt} is not a valid image link")
        if image is not None:
            if link: st.markdown(link, unsafe_allow_html=True)
            image = normalize(image)
            fig2d = generate_image_figure(image, dxy=1.0, ctf_type=None, title="Original Image", plot2d_s2=False, show_color=show_color)
            st.bokeh_chart(fig2d, use_container_width=True)

            fig2ds = []
            for i in range(n):
                _, _, ctf_2d = ctfs[i].ctf2d(apix, imagesize, over_sample, abs=plot_abs, plot_s2=False)
                image2 = np.abs(np.fft.ifft2(np.fft.fft2(image)*np.fft.fftshift(ctf_2d)))
                if n>1:
                    title = f"CTF Applied - {i+1}"
                else:
                    title = f"CTF Applied"
                fig2d = generate_image_figure(image2, dxy=1.0, ctf_type=None, title=title, plot2d_s2=False, show_color=show_color)
                fig2ds.append(fig2d)
            if len(fig2ds)>1:
                from bokeh.models import CrosshairTool
                crosshair = CrosshairTool(dimensions="both")
                crosshair.line_color = 'red'
                for fig in fig2ds: fig.add_tools(crosshair)
                from bokeh.layouts import gridplot
                figs_grid = gridplot(children=[fig2ds], toolbar_location=None)
                st.bokeh_chart(figs_grid, use_container_width=True)
            else:
                st.bokeh_chart(fig2d, use_container_width=True)

def generate_image_figure(image, dxy, ctf_type, title, plot2d_s2=False, show_color=False):
    w, h = image.shape
    tools = 'box_zoom,crosshair,pan,reset,save,wheel_zoom'
    from bokeh.plotting import figure
    fig2d = figure(frame_width=w, frame_height=h,
        x_range=(-w//2*dxy, (w//2-1)*dxy), y_range=(-h//2*dxy, (h//2-1)*dxy),
        tools=tools)
    fig2d.grid.visible = False
    fig2d.axis.visible = False
    fig2d.toolbar_location = None
    if title:
        fig2d.title.text = title
        fig2d.title.align = "center"
        fig2d.title.text_font_size = "18px"

    if ctf_type is not None:
        if plot2d_s2:
            source_data = dict(image=[image], x=[-w//2*dxy], y=[-h//2*dxy], dw=[w*dxy], dh=[h*dxy])
            tooltips = [
                ("Res", "@res Å"),
                ("s", "@s 1/Å"),
                ("s2", "@s2 1/Å^2"),
                ('angle', '@ang °'),
                (ctf_type, '@image')
            ]
        else:
            source_data = dict(image=[image], x=[-w//2*dxy], y=[-h//2*dxy], dw=[w*dxy], dh=[h*dxy])
            tooltips = [
                ("Res", "@res Å"),
                ("s", "@s 1/Å"),
                ('angle', '@ang °'),
                (ctf_type, '@image')
            ]
    else:
        source_data = dict(image=[image], x=[-w//2*dxy], y=[-h//2*dxy], dw=[w*dxy], dh=[h*dxy])
        tooltips = [
            ("x", "$x Å"),
            ("y", "$y Å"),
            ("val", '@image')
        ]

    palette = "Spectral11" if show_color else "Greys256"    # "Viridis256"   
    fig2d_image = fig2d.image(source=source_data, image='image', palette=palette, x='x', y='y', dw='dw', dh='dh')
    
    from bokeh.models.tools import HoverTool
    image_hover = HoverTool(renderers=[fig2d_image], tooltips=tooltips)
    fig2d.add_tools(image_hover)

    if ctf_type is not None:
        # avoid the need for embedding res/s/s2 image -> smaller fig object and less data to transfer
        from bokeh.models import CustomJS
        from bokeh.events import MouseMove
        mousemove_callback_code = """
        var x = cb_obj.x
        var y = cb_obj.y
        var angle = Math.round(Math.atan2(y, x)*180./Math.PI * 100)/100
        console.log(x, y, angle)
        var s, res, s2
        if(s2) {
            s2 = Math.round(Math.hypot(x, y) * 1e3)/1e3
            s = Math.round(Math.sqrt(s2) * 1e3)/1e3
            res = Math.round(1./s * 100)/100
            hover.tooltips[0][1] = res.toString() + " Å"
            hover.tooltips[1][1] = s.toString() + " Å"
            hover.tooltips[2][1] = s2.toString() + " 1/Å^2"
            hover.tooltips[3][1] = angle.toString() + " °"
        } 
        else {
            s = Math.round(Math.hypot(x, y) * 1e3)/1e3
            res = Math.round(1./s * 100)/100
            hover.tooltips[0][1] = res.toString() + " Å"
            hover.tooltips[1][1] = s.toString() + " 1/Å"
            hover.tooltips[2][1] = angle.toString() + " °"
        }
        """
        mousemove_callback = CustomJS(args={"hover":fig2d.hover[0], "s2":plot2d_s2}, code=mousemove_callback_code)
        fig2d.js_on_event(MouseMove, mousemove_callback)
    
    return fig2d
class CTF:
    def __init__(self, voltage=300.0, cs=2.7, ampcontrast=10.0, defocus=0.5, dfdiff=0.0, dfang=0.0, phaseshift=0.0, bfactor=0.0, alpha=0.0, cc=2.7, dE=0.0, dI=0.0, dZ=0.0, dXY=0.0):
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

    @st.cache(persist=True, show_spinner=False)
    def ctf1d(self, apix, imagesize, over_sample, abs, plot_s2=False, defocus_override=None):
        defocus_final = defocus_override if defocus_override is not None else self.defocus
        s_nyquist = 1./(2*apix)
        if plot_s2:
            ds2 = s_nyquist*s_nyquist/(imagesize//2*over_sample)
            s2 = np.arange(imagesize//2*over_sample+1, dtype=np.float)*ds2
            s = np.sqrt(s2)
        else:
            ds = s_nyquist/(imagesize//2*over_sample)
            s = np.arange(imagesize//2*over_sample+1, dtype=np.float)*ds
            s2 = s*s
        wl = 12.2639 / np.sqrt(self.voltage * 1000.0 + 0.97845 * self.voltage * self.voltage)  # Angstrom
        wl3 = np.power(wl, 3)
        phaseshift = self.phaseshift * np.pi / 180.0 + np.arcsin(self.ampcontrast/100.)
        gamma =2*np.pi*(-0.5*defocus_final*1e4*wl*s2 + .25*self.cs*1e7*wl**3*s2**2) - phaseshift
        
        from scipy.special import j0, sinc
        env = np.ones_like(gamma)
        if self.bfactor: env *= np.exp(-self.bfactor*s2/4.0)
        if self.alpha: env *= np.exp(-np.power(np.pi*self.alpha*(1.0e7*self.cs*wl*wl*s*s*s-1e4*defocus_final*s), 2.0)*1e-6)
        if self.dE: env *= np.exp(-np.power(np.pi*self.cc*wl*s*s* self.dE/self.voltage, 2.0)/(16*np.log(2))*1e8)
        if self.dI: env *= np.exp(-np.power(np.pi*self.cc*wl*s*s* self.dI,              2.0)/(4*np.log(2))*1e2)
        if self.dZ: env *= j0(np.pi*self.dZ*wl*s*s)
        if self.dXY: env *= sinc(np.pi*self.dXY*s)

        ctf = np.sin(gamma) * env
        if abs>=2: ctf = ctf*ctf
        elif abs==1: ctf = np.abs(ctf)

        return s, s2, ctf

    @st.cache(persist=True, show_spinner=False)
    def ctf2d(self, apix, imagesize, over_sample, abs, plot_s2=False):    
        s_nyquist = 1./(2*apix)
        if plot_s2:
            ds = None
            ds2 = s_nyquist*s_nyquist/(imagesize//2*over_sample)
            sx2 = np.arange(-imagesize*over_sample//2, imagesize*over_sample//2) * ds2
            sy2 = np.arange(-imagesize*over_sample//2, imagesize*over_sample//2) * ds2
            sx2, sy2 = np.meshgrid(sx2, sy2, indexing='ij')
            theta = -np.arctan2(sy2, sx2)
            s2 = np.hypot(sx2, sy2)
            s = np.sqrt(s2)
        else:
            ds2 = None
            ds = s_nyquist/(imagesize//2*over_sample)
            sx = np.arange(-imagesize*over_sample//2, imagesize*over_sample//2) * ds
            sy = np.arange(-imagesize*over_sample//2, imagesize*over_sample//2) * ds
            sx, sy = np.meshgrid(sx, sy, indexing='ij')
            theta = -np.arctan2(sy, sx)
            s2 = sx*sx + sy*sy
            s = np.sqrt(s2)

        defocus2d = self.defocus + self.dfdiff/2*np.cos( 2*(theta-self.dfang*np.pi/180.))

        wl = 12.2639 / np.sqrt(self.voltage * 1000.0 + 0.97845 * self.voltage * self.voltage)  # Angstrom
        phaseshift = self.phaseshift * np.pi / 180.0 + np.arcsin(self.ampcontrast/100.)

        gamma =2*np.pi*(-0.5*defocus2d*1e4*wl*s2 + .25*self.cs*1e7*wl**3*s2**2) - phaseshift

        from scipy.special import j0, sinc
        env = np.ones_like(gamma)
        if self.bfactor: env *= np.exp(-self.bfactor*s2/4.0)
        if self.alpha: env *= np.exp(-np.power(np.pi*self.alpha*(1.0e7*self.cs*wl*wl*s*s*s-1e4*self.defocus*s), 2.0)*1e-6)
        if self.dE: env *= np.exp(-np.power(np.pi*self.cc*wl*s*s* self.dE/self.voltage, 2.0)/(16*np.log(2))*1e8)
        if self.dI: env *= np.exp(-np.power(np.pi*self.cc*wl*s*s* self.dI,              2.0)/(4*np.log(2))*1e2)
        if self.dZ: env *= j0(np.pi*self.dZ*wl*s*s)
        if self.dXY: env *= sinc(np.pi*self.dXY*s)

        ctf = np.sin(gamma) * env
        if abs>=2: ctf = ctf*ctf
        elif abs==1: ctf = np.abs(ctf)

        return ds, ds2, ctf

@st.cache(persist=True, show_spinner=False)
def compute_radial_profile(image):
    ny, nx = image.shape
    rmax = min(nx//2, ny//2)+1
    
    r = np.arange(0, rmax, 1, dtype=np.float32)
    theta = np.arange(0, 360, 1, dtype=np.float32) * np.pi/180.
    n_theta = len(theta)

    theta_grid, r_grid = np.meshgrid(theta, r, indexing='ij', copy=False)
    y_grid = ny//2 + r_grid * np.sin(theta_grid)
    x_grid = nx//2 + r_grid * np.cos(theta_grid)

    coords = np.vstack((y_grid.flatten(), x_grid.flatten()))

    from scipy.ndimage.interpolation import map_coordinates
    polar = map_coordinates(image, coords, order=1).reshape(r_grid.shape)

    rad_profile = polar.mean(axis=0)
    return rad_profile

@st.cache(persist=True, show_spinner=False)
def normalize(data, percentile=(0, 100)):
    p0, p1 = percentile
    vmin, vmax = sorted(np.percentile(data, (p0, p1)))
    data2 = (data-vmin)/(vmax-vmin)
    return data2

@st.cache(persist=True, show_spinner=False, ttl=24*60*60.) # refresh every day
def get_emdb_ids():
    try:
        import pandas as pd
        emdb_ids = pd.read_csv("https://wwwdev.ebi.ac.uk/emdb/api/search/*%20AND%20current_status:%22REL%22?wt=csv&download=true&fl=emdb_id")
        emdb_ids = list(emdb_ids.iloc[:,0].str.split('-', expand=True).iloc[:, 1].values)
    except:
        emdb_ids = []
    return emdb_ids

@st.cache(persist=True, show_spinner=False)
def get_emdb_image(emd_id, invert_contrast=-1, rgb2gray=True, output_shape=None):
    emdb_ids = get_emdb_ids()
    if emd_id in emdb_ids:
        url = f"https://www.ebi.ac.uk/pdbe/static/entry/EMD-{emd_id}/400_{emd_id}.gif"
        return get_image(url, invert_contrast, rgb2gray, output_shape)
    else:
        return None

@st.cache(persist=True, show_spinner=False)
def get_image(url, invert_contrast=-1, rgb2gray=True, output_shape=None):
    from skimage.io import imread
    try:
        image = imread( url, as_gray=rgb2gray)    # return: numpy array
    except:
        return None
    if output_shape:
        from skimage.transform import resize
        image = resize(image, output_shape=output_shape)
    vmin, vmax = np.percentile(image, (5, 95))
    image = (image-vmin)/(vmax-vmin)    # set to range [0, 1]
    if invert_contrast<0: # detect if the image contrast should be inverted (i.e. to make background black)
        edge_vals = np.mean([image[0, :].mean(), image[-1, :].mean(), image[:, 0].mean(), image[:, -1].mean()])
        invert_contrast = edge_vals>0.5
    if invert_contrast>0:
        image = -image + 1
    return image

@st.cache(persist=True, show_spinner=False)
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    import base64
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="ctf_curve_table.csv">Download the CTF data</a>'
    return href

@st.cache(persist=True, show_spinner=False)
def setup_anonymous_usage_tracking():
    try:
        import pathlib, stat
        index_file = pathlib.Path(st.__file__).parent / "static/index.html"
        index_file.chmod(stat.S_IRUSR|stat.S_IWUSR|stat.S_IRGRP|stat.S_IROTH)
        txt = index_file.read_text()
        if txt.find("gtag/js?")==-1:
            txt = txt.replace("<head>", '''<head><script async src="https://www.googletagmanager.com/gtag/js?id=G-YV3ZFR8VG6"></script><script>window.dataLayer = window.dataLayer || [];function gtag(){dataLayer.push(arguments);}gtag('js', new Date());gtag('config', 'G-YV3ZFR8VG6');</script>''')
            index_file.write_text(txt)
    except:
        pass

# adapted from https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
try:
    import streamlit.ReportThread as ReportThread
    from streamlit.server.Server import Server
except Exception:
    # Streamlit >= 0.65.0
    import streamlit.report_thread as ReportThread
    from streamlit.server.server import Server

class SessionState(object):
    def __init__(self, **kwargs):
        """A new SessionState object.

        Parameters
        ----------
        **kwargs : any
            Default values for the session state.

        Example
        -------
        >>> session_state = SessionState(user_name='', favorite_color='black')
        >>> session_state.user_name = 'Mary'
        ''
        >>> session_state.favorite_color
        'black'

        """
        for key, val in kwargs.items():
            setattr(self, key, val)


def SessionState_get(**kwargs):
    """Gets a SessionState object for the current session.

    Creates a new object if necessary.

    Parameters
    ----------
    **kwargs : any
        Default values you want to add to the session state, if we're creating a
        new one.

    Example
    -------
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    ''
    >>> session_state.user_name = 'Mary'
    >>> session_state.favorite_color
    'black'

    Since you set user_name above, next time your script runs this will be the
    result:
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    'Mary'

    """
    # Hack to get the session object from Streamlit.

    ctx = ReportThread.get_report_ctx()

    this_session = None

    current_server = Server.get_current()
    if hasattr(current_server, '_session_infos'):
        # Streamlit < 0.56
        session_infos = Server.get_current()._session_infos.values()
    else:
        session_infos = Server.get_current()._session_info_by_id.values()

    for session_info in session_infos:
        s = session_info.session
        if (
            # Streamlit < 0.54.0
            (hasattr(s, '_main_dg') and s._main_dg == ctx.main_dg)
            or
            # Streamlit >= 0.54.0
            (not hasattr(s, '_main_dg') and s.enqueue == ctx.enqueue)
            or
            # Streamlit >= 0.65.2
            (not hasattr(s, '_main_dg') and s._uploaded_file_mgr == ctx.uploaded_file_mgr)
        ):
            this_session = s

    if this_session is None:
        raise RuntimeError(
            "Oh noes. Couldn't get your Streamlit Session object. "
            'Are you doing something fancy with threads?')

    # Got the session object! Now let's attach some state into it.

    if not hasattr(this_session, '_custom_session_state'):
        this_session._custom_session_state = SessionState(**kwargs)

    return this_session._custom_session_state

if __name__ == "__main__":
    setup_anonymous_usage_tracking()
    main()
