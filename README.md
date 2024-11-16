# CTF Simulation
This Web app provides an interactive online simulation of the [contrast transfer function (CTF)](https://en.wikipedia.org/wiki/Contrast_transfer_function) of transmission electron microscopes.

Click one of the links ([Site 1](https://ctfsimulation.streamlit.app) | [Site 2](https://app.py.cafe/app/wjiang/ctfsimulation-streamlit) | [Site 3](https://app.py.cafe/app/wjiang/ctfsimulation-solara) | [Site 4](https://app.py.cafe/app/wjiang/ctfsimulation-shiny)) to explore the CTF function!

<a href="https://ctfsimulation.streamlit.app/?show_psf=1&show_2d=1&show_2d_right=1&simulate_ctf_effect=1&defocus=0.5&defocus=0.0&phaseshift=0.0&phaseshift=90.0&imagesize=300&imagesize=300"><img src="./ctf_simulation.png" style='width: 100%; object-fit: contain'></a>

---
You can also embed the Web app into your own Website using an [iframe](https://www.w3schools.com/tags/tag_iframe.asp), for example using the following code
```html
<iframe src="https://ctfsimulation.streamlit.app/?embedded=true" style='width: 100%; height: 740px; overflow: visible; margin: 0px; resize: both; border-style:none;'></iframe>
```
