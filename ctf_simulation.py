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
required_packages = "streamlit numpy scipy bokeh".split()
import_with_auto_install(required_packages)

import streamlit as st
import numpy as np

#from memory_profiler import profile
#@profile(precision=4)
def main():
    title = "CTF Simulation"
    st.set_page_config(page_title=title, layout="wide")

    session_state = st.session_state
    if not len(session_state):  # only run once at the start of the session
        st.elements.utils._shown_default_value_warning = True
        ctfs, plot_settings, embed = parse_query_parameters()
        session_state.plot_settings = plot_settings
        session_state.embed = embed
        set_session_state_from_ctfs(ctfs)
    plot_settings = session_state.plot_settings
    embed = session_state.embed
    ctfs = get_ctfs_from_session_state()

    if embed:
        col1, col2 = st.columns((1, 5))
        col_info = col2
    else:
        st.title(title)
        col1 = st.sidebar

    with col1:
        if embed:
            n = 1
        else:
            n = int(st.number_input('Number of CTFs', value=max(1, len(ctfs)), min_value=1, step=1))
        if len(ctfs)==0:
            ctfs = [ CTF() for i in range(n) ]
        elif n>len(ctfs):
            ctfs += [ ctfs[-1].copy() for i in range(n-len(ctfs)) ]
        else:
            ctfs = ctfs[:n]
        set_session_state_from_ctfs(ctfs)
        
        assert(n == len(ctfs))
        # make radio display horizontal
        st.markdown('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

        show_marker_empties = []
        for i in range(n):
            if n>1:
                expander = st.expander(label=f"CTF {i+1}", expanded=True if i==n-1 else False)
            else:
                import contextlib
                expander = contextlib.nullcontext()
            with expander:
                if not embed:
                    options = ('CTF', '|CTF|', 'CTF^2')
                    st.radio(label='CTF type', options=options, index=0, key=f"ctf_type_{i}")
                st.number_input('defocus (µm)', value=st.session_state[f"defocus_{i}"], min_value=0.0, step=0.1, format="%.5g", key=f"defocus_{i}")
                if embed:
                    rotavg = False
                else:
                    st.number_input('astigmatism mag (µm)', value=st.session_state[f"dfdiff_{i}"], min_value=0.0, step=0.01, format="%g", key=f"dfdiff_{i}")
                    if n==1 and ctfs[i].dfdiff:
                        value = ctfs[i].ctf_type == 2
                        rotavg = st.checkbox(label='plot rotational average', value=value)
                    else:
                        rotavg = False
                    st.number_input('astigmatism angle (°)', value=st.session_state[f"dfang_{i}"], min_value=0.0, max_value=360., step=1.0, format="%g", key=f"dfang_{i}")
                st.number_input('phase shift (°)', value=st.session_state[f"phaseshift_{i}"], min_value=0.0, max_value=360., step=1.0, format="%g", key=f"phaseshift_{i}")
                st.number_input('pixel size (Å/pixel)', value=st.session_state[f"apix_{i}"], min_value=0.1, step=0.01, format="%g", key=f"apix_{i}")
                st.number_input('voltage (kV)', value=st.session_state[f"voltage_{i}"], min_value=10., step=100., format="%g", key=f"voltage_{i}")
                st.number_input('cs (mm)', value=st.session_state[f"cs_{i}"], min_value=0.0, step=0.1, format="%g", key=f"cs_{i}")
                st.number_input('amplitude contrast (percent)', value=st.session_state[f"ampcontrast_{i}"], min_value=0.0, max_value=100., step=10.0, format="%g", key=f"ampcontrast_{i}")
                if not embed:
                    st.number_input('image size (pixel)', value=int(st.session_state[f"imagesize_{i}"]), min_value=16, max_value=4096, step=4, key=f"imagesize_{i}")
                    st.slider('over-sample (1x, 2x, 3x, etc)', value=int(st.session_state[f"over_sample_{i}"]), min_value=1, max_value=6, step=1, key=f"over_sample_{i}")
                
                    #with st.expander("envelope functions", expanded=False):
                    st.number_input('b-factor (Å^2)', value=st.session_state[f"bfactor_{i}"], min_value=0.0, step=10.0, format="%g", key=f"bfactor_{i}")
                    st.number_input('beam convergence semi-angle (mrad)', value=st.session_state[f"alpha_{i}"], min_value=0.0, step=0.05, format="%g", key=f"alpha_{i}")
                    dE = st.number_input('energy spread (eV)', value=st.session_state[f"dE_{i}"], min_value=0.0, step=0.2, format="%g", key=f"dE_{i}")
                    dI = st.number_input('objective lens current spread (ppm)', value=st.session_state[f"dI_{i}"], min_value=0.0, step=0.2, format="%g", key=f"dI_{i}")
                    if dE or dI:
                        st.number_input('cc (mm)', min_value=0.0, step=0.1, format="%g", key=f"cc_{i}")
                    st.number_input('sample vertical motion (Å)', value=st.session_state[f"dZ_{i}"], min_value=0.0, step=20.0, format="%g", key=f"dZ_{i}")
                    st.number_input('sample horizontal motion (Å)', value=st.session_state[f"dXY_{i}"], min_value=0.0, step=0.2, format="%g", key=f"dXY_{i}")

                show_marker_empties.append(st.empty())

        if embed:
            show_1d = True
            show_2d = False
            show_psf = False
            plot_s2 = False
            show_data = False
            share_url = False
        else:
            value = int(plot_settings.get("show_1d", 1))
            show_1d = st.checkbox('Show 1D CTF', value=value)
            value = int(plot_settings.get("show_2d", 0))
            show_2d = st.checkbox('Show 2D CTF', value=value)
            if show_1d:
                value = int(plot_settings.get("show_psf", 0))
                show_psf = st.checkbox('Show point spread function', value=value)
                for i in range(n):
                    show_marker_empties[i].checkbox(label='Show markers on CTF line plots', key=f"show_marker_{i}")
                value = int(plot_settings.get("plot_s2", 0))
                plot_s2 = st.checkbox(label='Plot s^2 as x-axis/radius', value=value)
                show_data = st.checkbox('Show CTF raw data', value=False)
            else:
                show_psf = False
                plot_s2 = False            
            share_url = st.checkbox('Show sharable URL', value=False, help="Include relevant parameters in the browser URL to allow you to share the URL and reproduce the plots")
            if share_url:
                set_query_parameters(ctfs, show_1d, show_2d, show_psf, plot_s2)
            else:
                st.experimental_set_query_params()
    ctfs = get_ctfs_from_session_state()
    ctf_labels = ctf_varying_parameter_labels(ctfs)

    if not embed:
        if show_1d:
            if show_2d:
                col2, col3 = st.columns((5, 2))
            else:
                col2, col3 = st.columns((1, 0.01))
            col_info = col2
        else:
            col2, col3 = st.columns((0.01, 1))
            col_info = col3

    if show_1d:
        with col2:
            from bokeh.plotting import figure
            from bokeh.models import LegendItem
            if plot_s2:
                x_label = "s^2 (1/Å^2)"
                hover_x_var = "s^2"
                hover_x_val = "$x 1/Å^2"
            else:
                x_label = "s (1/Å)"
                hover_x_var = "s"
                hover_x_val = "$x 1/Å"
            y_label = f"{' / '.join([ctf.ctf_type for ctf in ctfs])}"

            tools = 'box_zoom,crosshair,hover,pan,reset,save,wheel_zoom'
            hover_tips = [("Res", "@res Å"), (hover_x_var, hover_x_val), ("fval", "$y")]
            if n>1 or (n==1 and ctfs[0].dfdiff):
                hover_tips = [("CTF type", "@ctf_type"), ("Defocus", "@defocus µm")] + hover_tips
            fig = figure(title="", x_axis_label=x_label, y_axis_label=y_label, tools=tools, tooltips=hover_tips)
            fig.title.align = "center"
            fig.title.text_font_size = "18px"

            from bokeh.palettes import Category10
            colors = Category10[10]
            line_dashes = 'dashed solid dotted dotdash dashdot'.split()

            legends = []
            raw_data = []
            for i in range(n):
                label0 = ctf_labels[i]
                color = colors[ i % len(colors) ]
                if n==1 and ctfs[i].dfdiff:
                    defocuses = [ctfs[i].defocus - ctfs[i].dfdiff, ctfs[i].defocus, ctfs[i].defocus + ctfs[i].dfdiff]
                else:
                    defocuses = [ctfs[i].defocus]
                for di, defocus in enumerate(defocuses):
                    s, s2, ctf = ctfs[i].ctf1d(plot_s2, defocus_override=defocus)
                    x = s2 if plot_s2 else s
                    res = np.hstack(([1e6],  1/s[1:]))
                    source = dict(x=x, res=res, y=ctf)
                    if n>1 or (n==1 and ctfs[0].dfdiff):
                        source["ctf_type"] = [ctfs[i].ctf_type] * len(x)
                        source["defocus"] = [defocus] * len(x)
                    line_dash = line_dashes[di] if len(defocuses)>1 else "solid"
                    line_width = 2 if len(defocuses)==1 or di==1 else 1
                    line = fig.line(x='x', y='y', color=color, source=source, line_dash=line_dash, line_width=line_width)
                    if ctfs[i].show_marker:
                        fig.circle(x='x', y='y', color=color, source=source)
                    if len(defocuses)>1:
                        label = f"defocus={round(defocus, 4):g} µm"
                    else:
                        label = label0
                    legends.append(LegendItem(label=label, renderers=[line]))
                    if show_data:
                        if len(ctfs)>1 or len(defocuses)>1:
                            label = f"{y_label} ({label})"
                        else:
                            label = f"{y_label}"
                        raw_data.append((label, s, x, ctf))

                if n==1 and rotavg:
                    _, _, ctf_2d = ctfs[i].ctf2d(plot_s2)
                    rad_profile = compute_radial_profile(ctf_2d)
                    x = s2 if plot_s2 else s
                    source = dict(x=x, res=1/s, y=rad_profile)
                    line = fig.line(x='x', y='y', source=source, color='red', line_dash="solid", line_width=2)
                    label = f"defocus={ctfs[i].defocus}/dfdiff={ctfs[i].dfdiff}-rotavg"
                    legends.append(LegendItem(label=label, renderers=[line]))
                    if show_data:
                        label = f"{y_label} ({label})"
                        raw_data.append((label, s, x, rad_profile))

            fig.x_range.start = 0
            fig.x_range.end = source['x'][-1]
            fig.y_range.start = -1 if "CTF" in [ctf.ctf_type for ctf in ctfs] else 0
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
            del fig

            if not embed:
                if show_psf:
                    tools = 'box_zoom,crosshair,hover,pan,reset,save,wheel_zoom'
                    hover_tips = [("x", "$x Å"), (f"PSF", "$y")]
                    if n>1:
                        hover_tips = [("Defocus", "@defocus µm")] + hover_tips
                    fig = figure(title=f"Point Spread Function", x_axis_label="x (Å)", y_axis_label="PSF", tools=tools, tooltips=hover_tips)
                    fig.title.align = "center"
                    fig.title.text_font_size = "18px"     
                    legends = []           
                    for i in range(n):
                        x_psf, psf = ctfs[i].psf1d()
                        source = dict(x=x_psf, y=psf)
                        if n>1: source["defocus"] = [ctfs[i].defocus] * len(x_psf)
                        line = fig.line(x='x', y='y', source=source, line_width=2, color=colors[i%len(colors)])
                        if ctfs[i].show_marker:
                            fig.circle(x='x', y='y', color=color, source=source)
                        legends.append(LegendItem(label=ctf_labels[i], renderers=[line]))
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
                    st.text("") # workaround for a layout bug in streamlit 
                    st.bokeh_chart(fig, use_container_width=True)
                    del fig

                if show_data:
                    import pandas as pd
                    for i, (col3_label, s, x, ctf) in enumerate(raw_data):
                        columns = [x_label, "Res (Å)", col3_label]
                        maxlen = max(map(len, columns))
                        columns = [col.rjust(maxlen+10) for col in columns]
                        data = np.zeros((len(x), 3))
                        data[:,0] = x
                        data[:,1] = 1./s
                        data[:,2] = ctf
                        df = pd.DataFrame(data, columns=columns)
                        st.dataframe(df, width=None)
                        label = f"Download the data - {col3_label}"
                        st.markdown(get_table_download_link(df, label=label), unsafe_allow_html=True)

    if show_2d and not embed:
        with col3:
            st.text("") # workaround for a layout bug in streamlit 

            show_color = False

            fig2ds = []
            for i in range(n):
                ds, ds2, ctf_2d = ctfs[i].ctf2d(plot_s2)
                dxy = ds2 if plot_s2 else ds
                if n>1:
                    title = f"{i+1} - {ctfs[i].ctf_type}"
                else:
                    title = f"{ctfs[i].ctf_type}"
                fig2d = generate_image_figure(ctf_2d, dxy, ctfs[i].ctf_type, title, plot_s2, show_color)
                fig2ds.append(fig2d)
                del ctf_2d
            if len(fig2ds)>1:
                from bokeh.models import CrosshairTool
                crosshair = CrosshairTool(dimensions="both")
                crosshair.line_color = 'red'
                for fig in fig2ds: fig.add_tools(crosshair)
                from bokeh.layouts import gridplot
                figs_grid = gridplot(children=[fig2ds], toolbar_location=None)
                st.bokeh_chart(figs_grid, use_container_width=True)
                del figs_grid
            else:
                st.bokeh_chart(fig2d, use_container_width=True)
                del fig2d
            
            with st.expander("Simulate the CTF effect"):
                input_modes = ["Delta Function"]
                emdb_ids = get_emdb_ids()
                input_modes += ["Random EMDB ID", "Input an EMDB ID"]
                if "emd_id" not in session_state:
                    import random
                    session_state.emd_id = random.choice(emdb_ids)
                input_modes += ["Input an image url"]
                input_mode = st.radio(label="Choose an input mode:", options=input_modes, index=3)
                if input_mode == "Random EMDB ID":
                    button_clicked = st.button(label="Change EMDB ID")
                    if button_clicked:
                        import random
                        session_state.emd_id = random.choice(emdb_ids)
                    input_txt = f"EMD-{session_state.emd_id}"
                elif input_mode == "Input an EMDB ID":
                    emd_id = session_state.emd_id        
                    label = "Input an EMDB ID"
                    value = f"EMD-{emd_id}"
                    input_txt = st.text_input(label=label, value=value).strip()
                    session_state.emd_id = input_txt.lower().split("emd_")[-1]
                elif input_mode == "Input an image url":
                    label = "Input an image url:"
                    value = "https://upload.wikimedia.org/wikipedia/commons/d/d3/Albert_Einstein_Head.jpg"
                    input_txt = st.text_input(label=label, value=value).strip()
            
            image = None
            link = None
            if input_mode == "Delta Function":
                nx = ctfs[0].imagesize * ctfs[0].over_sample
                ny = nx
                image = np.zeros((ny, nx), dtype=np.float32)
                image[ny//2, nx//2] = 255
            elif emdb_ids and input_txt.startswith("EMD-"):
                emd_id = input_txt[4:]
                if emd_id in emdb_ids:
                    session_state.emd_id = emd_id
                    image = get_emdb_image(emd_id, invert_contrast=-1, rgb2gray=True, output_shape=(ctfs[0].imagesize*ctfs[0].over_sample, ctfs[0].imagesize*ctfs[0].over_sample))
                    link = f'[EMD-{emd_id}](https://www.emdataresource.org/EMD-{emd_id})'
                else:
                    emd_id_bad = emd_id
                    emd_id = random.choice(emdb_ids)
                    st.warning(f"EMD-{emd_id_bad} does not exist. Please input a valid id (for example, a randomly selected valid id {emd_id})")
            elif input_txt.startswith("http") or input_txt.startswith("ftp"):   # input is a url
                url = input_txt
                image = get_image(url, invert_contrast=0, rgb2gray=True, output_shape=(ctfs[0].imagesize*ctfs[0].over_sample, ctfs[0].imagesize*ctfs[0].over_sample))
                if image is not None:
                    image = image[::-1, :]
                    link = f'[Image Link]({url})'
                else:
                    st.warning(f"{url} is not a valid image link")
            elif len(input_txt):
                st.warning(f"{input_txt} is not a valid image link")
            if image is not None:
                import_with_auto_install(["skimage:scikit_image"])
                if link: st.markdown(link, unsafe_allow_html=True)
                image = normalize(image)
                fig2d = generate_image_figure(image, dxy=1.0, ctf_type=None, title="Original Image", plot_s2=False, show_color=show_color)
                st.bokeh_chart(fig2d, use_container_width=True)
                del fig2d

                fig2ds = []
                for i in range(n):
                    _, _, ctf_2d = ctfs[i].ctf2d(plot_s2=False)
                    from skimage.transform import resize
                    image_work = resize(image, (ctfs[i].imagesize*ctfs[i].over_sample, ctfs[i].imagesize*ctfs[i].over_sample), anti_aliasing=True)
                    image2 = np.abs(np.fft.ifft2(np.fft.fft2(image_work)*np.fft.fftshift(ctf_2d)))
                    if n>1:
                        title = f"{i+1} - {ctfs[i].ctf_type} Applied"
                    else:
                        title = f"{ctfs[i].ctf_type} Applied"
                    fig2d = generate_image_figure(image2, dxy=1.0, ctf_type=None, title=title, plot_s2=False, show_color=show_color)
                    fig2ds.append(fig2d)
                    del ctf_2d, image2, image_work
                del image
                if len(fig2ds)>1:
                    from bokeh.models import CrosshairTool
                    crosshair = CrosshairTool(dimensions="both")
                    crosshair.line_color = 'red'
                    for fig in fig2ds: fig.add_tools(crosshair)
                    from bokeh.layouts import gridplot
                    figs_grid = gridplot(children=[fig2ds], toolbar_location=None)
                    st.bokeh_chart(figs_grid, use_container_width=True)
                    del figs_grid
                else:
                    st.bokeh_chart(fig2d, use_container_width=True)
                    del fig2d

    with col_info:
        if not embed:
            st.markdown("**Learn more about [Contrast Transfer Function (CTF)](https://en.wikipedia.org/wiki/Contrast_transfer_function):**\n* [Getting Started in CryoEM, Grant Jensen](https://www.youtube.com/watch?v=mPynoF2j6zc&t=2s)\n* [CryoEM Principles, Fred Sigworth](https://www.youtube.com/watch?v=Y8wivQTJEHQ&list=PLRqNpJmSRfar_z87-oa5W421_HP1ScB25&index=5)\n")

        st.markdown("*Developed by the [Jiang Lab@Purdue University](https://jiang.bio.purdue.edu/ctf-simulation). Report problems to Wen Jiang (jiang12 at purdue.edu)*")

    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def generate_image_figure(image, dxy, ctf_type, title, plot_s2=False, show_color=False):
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
        if plot_s2:
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
        mousemove_callback = CustomJS(args={"hover":fig2d.hover[0], "s2":plot_s2}, code=mousemove_callback_code)
        fig2d.js_on_event(MouseMove, mousemove_callback)
    
    return fig2d

def set_session_state_from_ctfs(ctfs, remove_extra_keys=True):
    if len(ctfs)==0: return
    n = len(ctfs)
    for i in range(n):
        d = ctfs[i].get_dict()
        for attr in d.keys():
            attr_i = f"{attr}_{i}"
            if attr_i not in st.session_state:
                st.session_state[attr_i] = d[attr]
    if remove_extra_keys:
        for k in st.session_state:
            if k.rfind("_")==-1: continue
            attr, i = k.rsplit("_", maxsplit=1)
            if attr in d and int(i)>=n:
                del st.session_state[k]

def get_ctfs_from_session_state():
    d = CTF().get_dict()
    attrs = []
    for k in st.session_state:
        if k.rfind("_")==-1: continue
        attr, i = k.rsplit("_", maxsplit=1)
        if attr in d:
            i = int(i)
            attrs.append( (i, attr, st.session_state[k]) )
    if len(attrs)<1:
        return []
    
    attrs.sort()
    n = attrs[-1][0]+1
    ctfs = [CTF() for i in range(n)]
    for i, attr, val in attrs:
        setattr(ctfs[i], attr, val)
    return ctfs

def set_query_parameters(ctfs, show_1d, show_2d, show_psf, plot_s2):
    d = {}
    if not show_1d: d["show_1d"] = 0
    if show_2d: d["show_2d"] = 1
    if show_psf: d["show_psf"] = 1
    if plot_s2: d["plot_s2"] = 1
    default_vals = CTF().get_dict()
    for attr in default_vals.keys():
        if attr == "ctf_type":
            vals = [getattr(ctfs[i], attr) for i in range(len(ctfs))]
            vals_set = set(vals)
            if len(vals_set)>1 or list(vals_set)[0] != "CTF":
                d[attr] = vals
        else:
            vals = np.array([getattr(ctfs[i], attr) for i in range(len(ctfs))])
            if np.any(vals - default_vals[attr]):
                d[attr] = vals
    st.experimental_set_query_params(**d)

def parse_query_parameters():
    query_params = st.experimental_get_query_params()
    embed = "embed" in query_params and query_params["embed"][0]!='0'
    attrs = CTF().get_dict().keys()
    ns = [len(query_params[attr]) for attr in attrs if attr in query_params]
    if not ns:
        ctfs = []
    else:
        n = int(max(ns))
        ctfs = [CTF() for i in range(n)]
        str_types = ["ctf_type"]
        int_types = ["imagesize", "over_sample", "show_marker"]
        for attr in attrs:
            if attr in query_params:
                for i in range(len(query_params[attr])):
                    if attr in str_types:
                        setattr(ctfs[i], attr, query_params[attr][i])
                    elif attr in int_types:
                        setattr(ctfs[i], attr, int(query_params[attr][i]))
                    else:
                        setattr(ctfs[i], attr, float(query_params[attr][i]))
    attrs = "show_1d show_2d show_psf plot_s2".split()
    plot_settings = {}
    for attr in attrs:
        if attr in query_params:
            plot_settings[attr] = int(query_params[attr][0])
    return ctfs, plot_settings, embed

def ctf_varying_parameter_labels(ctfs):
    str_types = ["ctf_type"]
    int_types = ["imagesize", "over_sample", "show_marker"]
    ret = []
    attrs = ctf_varying_parameters(ctfs)
    if attrs:
        for ctf in ctfs:
            tokens = []
            for attr in attrs:
                if attr in str_types:
                    tokens += [f'{attr}={getattr(ctf, attr)}']
                elif attr in int_types:
                    tokens += [f'{attr}={int(getattr(ctf, attr))}']
                else:
                    tokens += [f'{attr}={getattr(ctf, attr):.6g}']
            s = '/'.join(tokens)
            ret.append(s)
    else:
        ret = [f'{i+1}' for i in range(len(ctfs))]
    return ret

def ctf_varying_parameters(ctfs):
    if len(ctfs)<2: return []
    attrs = "voltage cs ampcontrast defocus dfdiff dfang phaseshift bfactor alpha cc dE dI dZ dXY apix imagesize over_sample ctf_type show_marker".split()
    ret = []
    for attr in attrs:
        vals = [getattr(ctfs[i], attr) for i in range(len(ctfs))]
        if attr in ["ctf_type"]:
            if len(set(vals))>1:
                ret.append(attr)
        else:
            vals = np.array(vals)
            if np.std(vals)>1e-6:
                ret.append(attr)
    return ret

class CTF:
    def __init__(self, voltage=300.0, cs=2.7, ampcontrast=7.0, defocus=0.5, dfdiff=0.0, dfang=0.0, phaseshift=0.0, bfactor=0.0, alpha=0.0, cc=2.7, dE=0.0, dI=0.0, dZ=0.0, dXY=0.0, apix=1.0, imagesize=256, over_sample=1, ctf_type='CTF', show_marker=0):
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
        self.ctf_type = ctf_type    # CTF, |CTF|, CTF^2
        self.show_marker = int(show_marker)
    
    def __str__(self):
        return str(self.get_dict())

    def __repr__(self):
        return self.__str__()

    def copy(self):
        import copy
        return copy.copy(self)
    
    def get_dict(self):
        ret = {}
        ret.update(self.__dict__)
        return ret

    #@st.cache(persist=True, show_spinner=False)
    def ctf1d(self, plot_s2=False, defocus_override=None):
        defocus_final = defocus_override if defocus_override is not None else self.defocus
        s_nyquist = 1./(2*self.apix)
        if plot_s2:
            ds2 = s_nyquist*s_nyquist/(self.imagesize//2*self.over_sample)
            s2 = np.arange(self.imagesize//2*self.over_sample+1, dtype=np.float32)*ds2
            s = np.sqrt(s2)
        else:
            ds = s_nyquist/(self.imagesize//2*self.over_sample)
            s = np.arange(self.imagesize//2*self.over_sample+1, dtype=np.float32)*ds
            s2 = s*s
        wl = 12.2639 / np.sqrt(self.voltage * 1000.0 + 0.97845 * self.voltage * self.voltage)  # Angstrom
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
        if self.ctf_type == "CTF^2": ctf = ctf*ctf
        elif self.ctf_type == "|CTF|": ctf = np.abs(ctf)

        return s, s2, ctf

    #@st.cache(persist=True, show_spinner=False)
    def psf1d(self, defocus_override=None):
        defocus_final = defocus_override if defocus_override is not None else self.defocus
        s_nyquist = 1./(2*self.apix)
        ds = s_nyquist/(self.imagesize//2)
        s = (np.arange(self.imagesize, dtype=np.float32) - self.imagesize//2)*ds
        s2 = s*s
        wl = 12.2639 / np.sqrt(self.voltage * 1000.0 + 0.97845 * self.voltage * self.voltage)  # Angstrom
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
        if self.ctf_type == "CTF^2": ctf = ctf*ctf
        elif self.ctf_type== "|CTF|": ctf = np.abs(ctf)

        unity = np.ones((self.imagesize,), dtype=np.complex64)
        psf = np.abs( np.fft.ifft( unity * np.fft.ifftshift(ctf) ) )
        psf = np.fft.fftshift(psf)
        psf /= np.linalg.norm(psf, ord=2)
        x = (np.arange(self.imagesize)-self.imagesize//2) * self.apix

        return x, psf

    #@st.cache(persist=True, show_spinner=False)
    def ctf2d(self, plot_s2=False):    
        s_nyquist = 1./(2*self.apix)
        if plot_s2:
            ds = None
            ds2 = s_nyquist*s_nyquist/(self.imagesize//2*self.over_sample)
            sx2 = np.arange(-self.imagesize*self.over_sample//2, self.imagesize*self.over_sample//2) * ds2
            sy2 = np.arange(-self.imagesize*self.over_sample//2, self.imagesize*self.over_sample//2) * ds2
            sx2, sy2 = np.meshgrid(sx2, sy2, indexing='ij')
            theta = -np.arctan2(sy2, sx2)
            s2 = np.hypot(sx2, sy2)
            s = np.sqrt(s2)
            del sx2, sy2
        else:
            ds2 = None
            ds = s_nyquist/(self.imagesize//2*self.over_sample)
            sx = np.arange(-self.imagesize*self.over_sample//2, self.imagesize*self.over_sample//2) * ds
            sy = np.arange(-self.imagesize*self.over_sample//2, self.imagesize*self.over_sample//2) * ds
            sx, sy = np.meshgrid(sx, sy, indexing='ij')
            theta = -np.arctan2(sy, sx)
            s2 = sx*sx + sy*sy
            s = np.sqrt(s2)
            del sx, sy

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
        if self.ctf_type == "CTF^2": ctf = ctf*ctf
        elif self.ctf_type== "|CTF|": ctf = np.abs(ctf)
        del gamma, env, phaseshift, defocus2d, s, s2, theta
        return ds, ds2, ctf

#@st.cache(max_entries=10, ttl=3600, persist=True, show_spinner=False)
def compute_radial_profile(image):
    ny, nx = image.shape
    rmax = min(nx//2, ny//2)+1
    
    r = np.arange(0, rmax, 1, dtype=np.float32)
    theta = np.arange(0, 360, 1, dtype=np.float32) * np.pi/180.

    theta_grid, r_grid = np.meshgrid(theta, r, indexing='ij', copy=False)
    y_grid = ny//2 + r_grid * np.sin(theta_grid)
    x_grid = nx//2 + r_grid * np.cos(theta_grid)

    coords = np.vstack((y_grid.flatten(), x_grid.flatten()))

    from scipy.ndimage.interpolation import map_coordinates
    polar = map_coordinates(image, coords, order=1).reshape(r_grid.shape)

    rad_profile = polar.mean(axis=0)
    return rad_profile

#@st.cache(persist=True, show_spinner=False)
def normalize(data, percentile=(0, 100)):
    p0, p1 = percentile
    vmin, vmax = sorted(np.percentile(data, (p0, p1)))
    data2 = (data-vmin)/(vmax-vmin)
    return data2.astype(np.float32)

@st.cache(persist=True, show_spinner=False, ttl=24*60*60.) # refresh every day
def get_emdb_ids():
    try:
        import pandas as pd
        emdb_ids = pd.read_csv("https://wwwdev.ebi.ac.uk/emdb/api/search/*%20AND%20current_status:%22REL%22?wt=csv&download=true&fl=emdb_id")
        emdb_ids = list(emdb_ids.iloc[:,0].str.split('-', expand=True).iloc[:, 1].values)
    except:
        emdb_ids = ["11638"]
    return emdb_ids

@st.cache(max_entries=10, ttl=3600, persist=True, show_spinner=False)
def get_emdb_image(emd_id, invert_contrast=-1, rgb2gray=True, output_shape=None):
    emdb_ids = get_emdb_ids()
    if emd_id in emdb_ids:
        url = f"https://www.ebi.ac.uk/emdb/images/entry/EMD-{emd_id}/400_{emd_id}.gif"
        return get_image(url, invert_contrast, rgb2gray, output_shape)
    else:
        return None

@st.cache(max_entries=10, ttl=3600, persist=True, show_spinner=False)
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

#@st.cache(persist=True, show_spinner=False)
def get_table_download_link(df, label="Download the CTF data"):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    import base64
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="ctf_curve_table.csv">{label}</a>'
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

def print_memory_usage():
    from inspect import currentframe
    import psutil, os
    cf = currentframe()
    print(f'Line {cf.f_back.f_lineno}: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MB')

if __name__ == "__main__":
    setup_anonymous_usage_tracking()
    main()
