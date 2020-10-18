import streamlit as st
import numpy as np
import pandas as pd
import random

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

@st.cache()
def get_emdb_ids():
    emdb_ids = pd.read_csv("https://wwwdev.ebi.ac.uk/pdbe/emdb/emdb_schema3/api/search/*%20AND%20current_status:%22REL%22?wt=csv&download=true&fl=emdb_id")
    emdb_ids = list(emdb_ids.iloc[:,0].str.split('-', expand=True).iloc[:, 1].values)
    return emdb_ids

@st.cache()
def get_emdb_image(emd_id, invert_contrast=-1, rgb2gray=True, output_shape=None):
    url = f"https://www.ebi.ac.uk/pdbe/static/entry/EMD-{emd_id}/400_{emd_id}.gif"
    from skimage.io import imread
    image = imread( url )    # numpy array
    if rgb2gray:
        from skimage.color import rgb2gray
        image = rgb2gray(image)
    if output_shape:
        from skimage.transform import resize
        image = resize(image, output_shape=output_shape)
    if invert_contrast<0: # detect if the image contrast should be inverted (i.e. to make background black)
        from scipy.stats import skew
        invert_contrast = skew(image, axis=None)>0
    if invert_contrast>0:
        vmin, vmax = image.min(), image.max()
        image = -image + vmax + vmin
    return image

session_state = SessionState_get(defocus=0.5, emd_id=0)

st.beta_set_page_config(page_title="CTF Simulation", layout="wide")

with st.sidebar:
    session_state.defocus = st.number_input('defocus (micrometer)', value=session_state.defocus)
    session_state.defocus = st.slider('', min_value=0.0, max_value=10.0, value=session_state.defocus, step=0.0001)
    dfdiff = st.number_input('astigmatism mag (micrometer)', value=0.0)
    dfang = st.number_input('astigmatism angle (degree)', value=0.0, min_value=0.0, max_value=360.)
    phaseshift = st.number_input('phase shift (degree)', value=0.0)
    apix = st.number_input('pixel size (Angstrom/pixel)', value=1.0)
    imagesize = st.number_input('image size (pixel)', value=256, min_value=32, max_value=4096)
    bfactor = st.number_input('b-factor (Angstrom^2)', value=0.0)
    voltage = st.number_input('voltage (kV)', value=300)
    cs = st.number_input('cs (mm)', value=2.7)
    ampcontrast = st.number_input('ampltude contrast (percent)', value=10., min_value=0.0, max_value=100.)

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
    if abs: ctf = np.abs(ctf)

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
    if abs: ctf = np.abs(ctf)

    if plot_s2:
        s2 = np.sqrt(sx*sx + sy*sy)
        gamma =2*np.pi*(-0.5*defocus2d*1e4*wl*s2 + .25*cs*1e7*wl**3*s2**2) - phaseshift
        ctf_s2 = np.sin(gamma) * np.exp(-bfactor*s2/4.0) 
        if abs: ctf_s2 = np.abs(ctf_s2)
    else:
        ctf_s2 = None
    return ctf, ctf_s2

st.title("CTF Simulation")
col1, _, col2 = st.beta_columns((3, 0.1, 2))
over_sample = col1.slider('over-sample (1x, 2x, 3x, etc)', value=1, min_value=1, max_value=6)
plot_abs = col2.checkbox("plot amplitude", value=False)

col1d, _, col2d = st.beta_columns((3, 0.1, 2))
with col1d:
    label = 'plot s^2 as x-axis'
    plot1d_s2 = st.checkbox(label, value=False)

    s, s2, ctf = ctf1d(voltage, cs, ampcontrast, session_state.defocus, phaseshift, bfactor, apix, imagesize, over_sample, plot_abs)

    from bokeh.plotting import figure
    if plot1d_s2:
        x = s2
        x_label = "s^2 (1/Angstrom^2)"
    else:
        x = s
        x_label = "s (1/Angstrom)"
    if plot_abs:
        y_label = "|Contrast Transfer Function|"
    else:
        y_label = "Contrast Transfer Function"
    tools = 'box_zoom,crosshair,hover,pan,reset,save,wheel_zoom'
    fig = figure(title="", x_axis_label=x_label, y_axis_label=y_label, x_range=(0, x[-1]), tools=tools)
    fig.line(x=x, y=ctf, line_width=2)
    st.bokeh_chart(fig, use_container_width=True)

    show_data = st.checkbox('show raw data', value=False)
    if show_data:
        data = np.zeros((len(x), 3))
        data[:,0] = x
        data[:,1] = 1./s
        data[:,2] = ctf
        df = pd.DataFrame(data, columns=(x_label, "resolution (Angstrom)", y_label))
        st.dataframe(df, width=900)

    st.markdown("*Developed by the [Jiang Lab@Purdue University](https://jiang.bio.purdue.edu). Report problems to Wen Jiang (jiang12 at purdue.edu)*")

with col2d:
    label = 'plot s^2 as radius'
    plot2d_s2 = st.checkbox(label, value=False)

    ctf, ctf_s2 = ctf2d(voltage, cs, ampcontrast, session_state.defocus, dfdiff, dfang, phaseshift, bfactor, apix, imagesize, over_sample, plot_abs, plot2d_s2)
    ctf_to_plot = ctf_s2 if ctf_s2 is not None else ctf
    st.image(ctf_to_plot, clamp=[ctf.min(), ctf.max()])

    emdb_ids = get_emdb_ids()
    if session_state.emd_id==0:
        emd_id = random.choice(emdb_ids)
    else:
        emd_id = session_state.emd_id
    with st.beta_expander("Simulate the CTF effect on an image"):
        emd_id = st.text_input('Input an EMDB ID:', value=emd_id)
    if emd_id in emdb_ids:
        session_state.emd_id = emd_id
        image = get_emdb_image(emd_id, invert_contrast=-1, rgb2gray=True, output_shape=(imagesize*over_sample, imagesize*over_sample))
        link = f'[EMD-{emd_id}](https://www.ebi.ac.uk/pdbe/entry/emdb/EMD-{emd_id})'
        st.markdown(link, unsafe_allow_html=True)
        st.image(image, caption="Orignal image", clamp=[image.min(), image.max()])
        # apply ctf to the image
        image2 = np.abs(np.fft.ifft2(np.fft.fft2(image)*np.fft.fftshift(ctf)))
        st.image(image2, caption="CTF applied", clamp=[image2.min(), image2.max()])
    else:
        emd_id_bad = emd_id
        emd_id = random.choice(emdb_ids)
        st.warning(f"EMD-{emd_id_bad} does not exist. Please input a valid id (for example, a randomly selected valid id {emd_id})")

