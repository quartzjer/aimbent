import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';

export default class MyExtension extends Extension {
    constructor(metadata) {
        super(metadata);
    }

    enable() {
        // Called when user (or GNOME) enables the extension
        log(`Enabling ${this.metadata.name}`);
        global.context.unsafe_mode = true;
    }

    disable() {
        // Called when user (or GNOME) disables the extension
        log(`Disabling ${this.metadata.name}`);
        global.context.unsafe_mode = false;
    }
}
