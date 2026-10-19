// viewlight.js -- STUB. PD walls face-on, time strip, playback; replaced by the light work package.
// It drives the 3-D PD wall colours with app.views.view3d.setPdColors(Float32Array rgb)
// (n_tiles*3, 0..1 sRGB, tile order = geometry 'pd/xyz'); call view3d.setPdColors(null) to
// return the walls to their neutral colour. See the interface comment at the top of main.js.
export default {
  name: 'viewlight',
  init(container, app) {
    this.container = container;
    container.innerHTML = '<div class="hud-title">Light view (not implemented yet)</div>';
  },
  setGeometry(geometry) {},
  setEvent(bundle, geometry) {},
  onState(state, changed) {},
  setVisible(visible) {},
  resize() {},
  getCanvas() { return null; },
};
