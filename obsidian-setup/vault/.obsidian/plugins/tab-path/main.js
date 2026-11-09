const { Plugin } = require("obsidian");

const DEPTH = 1; // how many parent folders to show before the file name

module.exports = class TabPath extends Plugin {
  onload() {
    const refresh = () => this.app.workspace.iterateAllLeaves((leaf) => this.label(leaf));
    this.registerEvent(this.app.workspace.on("layout-change", refresh));
    this.registerEvent(this.app.workspace.on("file-open", refresh));
    this.registerEvent(this.app.vault.on("rename", refresh));
    this.app.workspace.onLayoutReady(refresh);
  }

  label(leaf) {
    const file = leaf.view && leaf.view.file;
    const el = leaf.tabHeaderInnerTitleEl;
    if (!file || !el) return;
    const parts = file.path.split("/");
    parts[parts.length - 1] = file.basename;
    el.setText(parts.slice(-1 - DEPTH).join("/"));
    leaf.tabHeaderEl && leaf.tabHeaderEl.setAttribute("aria-label", file.path);
  }

  onunload() {
    this.app.workspace.iterateAllLeaves((leaf) => leaf.updateHeader && leaf.updateHeader());
  }
};
