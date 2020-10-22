import streamlit as st
import numpy as np

def main():
    session_state = SessionState_get(defocus=0.5, emd_id=0)

    st.beta_set_page_config(page_title="CTF Simulation", layout="wide")

    with st.sidebar:    # sidebar at the left of the screen
        options = ('CTF', '|CTF|', 'CTF^2')
        ctf_type = st.selectbox(label='CTF type', options=options)
        plot_abs = options.index(ctf_type)
        session_state.defocus = st.number_input('defocus (micrometer)', min_value=0.0, max_value=10.0, value=session_state.defocus, step=0.01, format="%.4f")
        session_state.defocus = st.slider('', min_value=0.0, max_value=10.0, value=session_state.defocus, step=0.01, format="%.4f")
        dfdiff = st.number_input('astigmatism mag (micrometer)', value=0.0, min_value=0.0, max_value=session_state.defocus, step=0.01, format="%.4f")
        dfang = st.number_input('astigmatism angle (degree)', value=0.0, min_value=0.0, max_value=360., step=1.0)
        phaseshift = st.number_input('phase shift (degree)', value=0.0, min_value=0.0, max_value=360., step=1.0)
        apix = st.number_input('pixel size (Angstrom/pixel)', value=1.0, min_value=0.1, max_value=10., step=0.01)
        imagesize = st.number_input('image size (pixel)', value=256, min_value=32, max_value=4096, step=4)
        over_sample = st.slider('over-sample (1x, 2x, 3x, etc)', value=1, min_value=1, max_value=6, step=1)
        bfactor = st.number_input('b-factor (Angstrom^2)', value=0.0, min_value=0.0, max_value=1000.0, step=10.0)
        voltage = st.number_input('voltage (kV)', value=300, min_value=10, max_value=3000, step=100)
        cs = st.number_input('cs (mm)', value=2.7, min_value=0.0, max_value=10.0, step=0.1)
        ampcontrast = st.number_input('ampltude contrast (percent)', value=10., min_value=0.0, max_value=100., step=10.0)

    # right-side main panel
    st.title("CTF Simulation")
    col1d, _, col2d = st.beta_columns((3, 0.1, 2))
    with col1d: # left side column
        label = 'plot s^2 as x-axis'
        plot1d_s2 = st.checkbox(label, value=False)

        s, s2, ctf = ctf1d(voltage, cs, ampcontrast, session_state.defocus, phaseshift, bfactor, apix, imagesize, over_sample, plot_abs)

        from bokeh.plotting import figure, ColumnDataSource
        if plot1d_s2:
            x = s2
            x_label = "s^2 (1/Angstrom^2)"
            hover_x_var = "s^2"
            hover_x_val = "$x 1/A^2"
        else:
            x = s
            x_label = "s (1/Angstrom)"
            hover_x_var = "s"
            hover_x_val = "$x 1/A"
        y_label = f"{ctf_type}"
        source = ColumnDataSource(data=dict(x=x, res=1/s, y=ctf))
        tools = 'box_zoom,crosshair,hover,pan,reset,save,wheel_zoom'
        hover_tips = [(hover_x_var, hover_x_val), ("Res", "@res A"), (f"{ctf_type}", "$y")]
        fig = figure(title="", x_axis_label=x_label, y_axis_label=y_label, x_range=(0, x[-1]), tools=tools, tooltips=hover_tips)
        fig.line(x='x', y='y', source=source, line_width=2)
        st.bokeh_chart(fig, use_container_width=True)

        show_data = st.checkbox('show raw data', value=False)
        if show_data:
            import pandas as pd
            data = np.zeros((len(x), 3))
            data[:,0] = x
            data[:,1] = 1./s
            data[:,2] = ctf
            df = pd.DataFrame(data, columns=(x_label.rjust(15), "Res (Angstrom)".rjust(15), y_label.rjust(15)))
            st.dataframe(df, width=600)

        st.markdown("*Developed by the [Jiang Lab@Purdue University](https://jiang.bio.purdue.edu). Report problems to Wen Jiang (jiang12 at purdue.edu)*")

    with col2d: # right-side column
        label = 'plot s^2 as radius'
        plot2d_s2 = st.checkbox(label, value=False)

        ctf, ctf_s2 = ctf2d(voltage, cs, ampcontrast, session_state.defocus, dfdiff, dfang, phaseshift, bfactor, apix, imagesize, over_sample, plot_abs, plot2d_s2)
        ctf_to_plot = ctf_s2 if ctf_s2 is not None else ctf
        st.image(ctf_to_plot, clamp=[ctf.min(), ctf.max()])

        import random
        emdb_ids = get_emdb_ids()
        if session_state.emd_id==0:
            emd_id = random.choice(emdb_ids)
        else:
            emd_id = session_state.emd_id
        with st.beta_expander("Simulate the CTF effect on an image"):
            input_txt = st.text_input('Input an EMDB ID or image url:', value=emd_id)
            input_txt=input_txt.strip()
        image = None
        link = None
        if input_txt.startswith("http") or input_txt.startswith("ftp"):   # input is a url
            url = input_txt
            image = get_image(url, invert_contrast=-1, rgb2gray=True, output_shape=(imagesize*over_sample, imagesize*over_sample))
            if image is not None:
                link = f'[Image Link]({url})'
            else:
                st.warning(f"{url} is not a valid image link")
        else:   # default to an EMDB ID
            emd_id = input_txt
            if emd_id in emdb_ids:
                session_state.emd_id = emd_id
                image = get_emdb_image(emd_id, invert_contrast=-1, rgb2gray=True, output_shape=(imagesize*over_sample, imagesize*over_sample))
                link = f'[EMD-{emd_id}](https://www.ebi.ac.uk/pdbe/entry/emdb/EMD-{emd_id})'
            else:
                emd_id_bad = emd_id
                emd_id = random.choice(emdb_ids)
                st.warning(f"EMD-{emd_id_bad} does not exist. Please input a valid id (for example, a randomly selected valid id {emd_id})")
        if image is not None:
            st.markdown(link, unsafe_allow_html=True)
            st.image(image, caption="Orignal image", clamp=[image.min(), image.max()])
            # apply ctf to the image
            image2 = np.abs(np.fft.ifft2(np.fft.fft2(image)*np.fft.fftshift(ctf)))
            st.image(image2, caption=f"{ctf_type} applied", clamp=[image2.min(), image2.max()])

@st.cache()
def ctf1d(voltage, cs, ampcontrast, defocus, phaseshift, bfactor, apix, imagesize, over_sample, abs):
    ds = 1./(apix*imagesize*over_sample)
    s = np.arange(imagesize*over_sample//2+1, dtype=np.float)*ds
    s2 = s*s
    wl = 12.2639 / np.sqrt(voltage * 1000.0 + 0.97845 * voltage * voltage)
    wl3 = np.power(wl, 3)
    phaseshift = phaseshift * np.pi / 180.0 + np.arcsin(ampcontrast/100.)
    gamma =2*np.pi*(-0.5*defocus*1e4*wl*s2 + .25*cs*1e7*wl**3*s2**2) - phaseshift
    ctf = np.sin(gamma) * np.exp(-bfactor*s2/4.0)
    if abs>=2: ctf = ctf*ctf
    elif abs==1: ctf = np.abs(ctf)

    return s, s2, ctf

@st.cache()
def ctf2d(voltage, cs, ampcontrast, defocus, dfdiff, dfang, phaseshift, bfactor, apix, imagesize, over_sample, abs, plot_s2=False):
    ds = 1./(apix*imagesize*over_sample)
    sx = np.arange(-imagesize*over_sample//2, imagesize*over_sample//2) * ds
    sy = np.arange(-imagesize*over_sample//2, imagesize*over_sample//2) * ds
    sx, sy = np.meshgrid(sx, sy, indexing='ij')

    theta = np.arctan2(sy, sx)
    defocus2d = defocus + dfdiff/2*np.cos( 2*(theta-dfang*np.pi/180.))

    wl = 12.2639 / np.sqrt(voltage * 1000.0 + 0.97845 * voltage * voltage)
    wl3 = np.power(wl, 3)
    phaseshift = phaseshift * np.pi / 180.0 + np.arcsin(ampcontrast/100.)

    s2 = sx*sx + sy*sy
    gamma =2*np.pi*(-0.5*defocus2d*1e4*wl*s2 + .25*cs*1e7*wl**3*s2**2) - phaseshift
    ctf = np.sin(gamma) * np.exp(-bfactor*s2/4.0)
    if abs>=2: ctf = ctf*ctf
    elif abs==1: ctf = np.abs(ctf)

    if plot_s2:
        ds2 = np.power(1./(2*apix), 2)/(imagesize*over_sample//2)
        s2x = np.arange(-imagesize*over_sample//2, imagesize*over_sample//2) * ds2
        s2y = np.arange(-imagesize*over_sample//2, imagesize*over_sample//2) * ds2
        s2x, s2y = np.meshgrid(s2x, s2y, indexing='ij')
        s2 = np.hypot(s2x, s2y)
        gamma =2*np.pi*(-0.5*defocus2d*1e4*wl*s2 + .25*cs*1e7*wl**3*s2**2) - phaseshift
        ctf_s2 = np.sin(gamma) * np.exp(-bfactor*s2/4.0) 
        if abs>=2: ctf_s2 = ctf_s2*ctf_s2
        elif abs==1: ctf_s2 = np.abs(ctf_s2)
    else:
        ctf_s2 = None
    return ctf, ctf_s2

@st.cache(persist=True, show_spinner=False, ttl=24*60*60.) # refresh every day
def get_emdb_ids():
    import pandas as pd
    emdb_ids = pd.read_csv("https://wwwdev.ebi.ac.uk/pdbe/emdb/emdb_schema3/api/search/*%20AND%20current_status:%22REL%22?wt=csv&download=true&fl=emdb_id")
    emdb_ids = list(emdb_ids.iloc[:,0].str.split('-', expand=True).iloc[:, 1].values)
    return emdb_ids

@st.cache()
def get_emdb_image(emd_id, invert_contrast=-1, rgb2gray=True, output_shape=None):
    emdb_ids = get_emdb_ids()
    if emd_id in emdb_ids:
        url = f"https://www.ebi.ac.uk/pdbe/static/entry/EMD-{emd_id}/400_{emd_id}.gif"
        return get_image(url, invert_contrast, rgb2gray, output_shape)
    else:
        return None

@st.cache()
def get_image(url, invert_contrast=-1, rgb2gray=True, output_shape=None):
    from skimage.io import imread
    try:
        image = imread( url, as_gray=rgb2gray)    # return: numpy array
    except:
        return None
    if output_shape:
        from skimage.transform import resize
        image = resize(image, output_shape=output_shape)
    vmin, vmax = image.min(), image.max()
    image = (image-vmin)/(vmax-vmin)    # set to range [0, 1]
    if invert_contrast<0: # detect if the image contrast should be inverted (i.e. to make background black)
        edge_vals = np.mean([image[0, :].mean(), image[-1, :].mean(), image[:, 0].mean(), image[:, -1].mean()])
        invert_contrast = edge_vals>0.5
    if invert_contrast>0:
        image = -image + 1
    return image


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

main()
