const { Plugin, WorkspaceLeaf } = require("obsidian");

const DEPTH = 1; // how many parent folders to show before the file name

module.exports = class TabPath extends Plugin {
  onload() {
    // Obsidian rewrites a tab title in updateHeader (open, rename, view change),
    // so relabel right after it instead of guessing which event will fire.
    const proto = WorkspaceLeaf.prototype;
    const original = proto.updateHeader;
    const label = (leaf) => this.label(leaf);
    proto.updateHeader = function (...args) {
      const result = original.apply(this, args);
      label(this);
      return result;
    };
    this.register(() => { proto.updateHeader = original; });
    this.app.workspace.onLayoutReady(() =>
      this.app.workspace.iterateAllLeaves((leaf) => this.label(leaf)));
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
