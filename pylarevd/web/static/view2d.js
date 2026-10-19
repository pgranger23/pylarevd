// view2d.js -- STUB. Wire-plane panels; replaced by the 2-D work package.
// Implements the view module interface documented at the top of main.js.
export default {
  name: 'view2d',
  init(container, app) {
    this.container = container;
    container.innerHTML = '<div class="hud-title">2D wire panels (not implemented yet)</div>';
  },
  setGeometry(geometry) {},
  setEvent(bundle, geometry) {},
  onState(state, changed) {},
  setVisible(visible) {},
  resize() {},
  getCanvas() { return null; },
};
