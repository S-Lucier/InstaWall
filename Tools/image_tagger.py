"""
Image Tagger Tool (Multi-Label Version)

Unlike the classifier which moves images to a single folder,
this tagger allows applying multiple tags to each image.
Tags are saved to a JSON file for later filtering.

Usage:
    python image_tagger.py --source ./images --output ./tagged_data

Controls:
    1-9, a-z: Toggle tag on/off for current image
    Enter: Confirm tags and move to next image
    Space: Skip (no tags)
    Backspace: Go back to previous image
    Escape: Save and quit
"""

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk

# Debug log file
DEBUG_LOG = Path(__file__).parent / "tagger_debug.log"
def debug(msg):
    with open(DEBUG_LOG, 'a') as f:
        f.write(str(msg) + '\n')
    print(msg)

debug("=== STARTING ===" )


class ImageTagger:
    """GUI application for tagging images with multiple labels."""

    def __init__(self, source_dir, output_file, tags=None, extensions=None,
                 skip_tagged=False, start_index=None):
        """
        Args:
            source_dir: Directory containing images to tag
            output_file: JSON file to save tags
            tags: Dict mapping keys to tag names
            extensions: Valid image extensions
            skip_tagged: If True, start at first untagged image
            start_index: Start at specific image number (1-indexed)
        """
        self.source_dir = Path(source_dir)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.extensions = extensions or {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}

        # Default tags for battlemap classification
        self.tags = tags or {
            # Map type
            '1': 'dungeon',
            '2': 'outdoor',
            '3': 'urban',
            '4': 'cave',
            '5': 'ship',
            # Quality
            'q': 'high_quality',
            'w': 'low_quality',
            # Grid
            'g': 'has_grid',
            # Variants
            'v': 'is_variant',
            # Use in training
            '0': 'keep',
            'd': 'discard',
        }

        # Collect images
        self.images = self._collect_images()
        self.current_index = 0
        self.current_tags = set()  # Tags for current image

        # Load existing tags or create new
        self.all_tags = self._load_tags()

        # Handle start position
        if start_index is not None:
            # Convert 1-indexed to 0-indexed
            self.current_index = max(0, min(start_index - 1, len(self.images) - 1))
            print(f"Starting at image {self.current_index + 1}")
        elif skip_tagged:
            self.current_index = self._find_first_untagged()
            print(f"Skipping to first untagged image: {self.current_index + 1}")

        # Setup GUI
        self.root = tk.Tk()
        self.root.title("Image Tagger")
        self._setup_gui()
        self._bind_keys()

        # Force window to render before loading images
        self.root.update()
        self.root.geometry("1000x800")
        self.root.update()

        # Load first image
        debug(f"Images list length: {len(self.images)}")
        debug(f"Current index: {self.current_index}")
        if self.images:
            debug("Calling _show_current_image...")
            self._show_current_image()
        else:
            debug("No images found!")
            self.status_label.config(text="No images found!")

    def _collect_images(self):
        """Collect all image files from source directory."""
        images = []

        debug(f"Searching in: {self.source_dir}")
        debug(f"Directory exists: {self.source_dir.exists()}")

        # List all files and filter by extension (case-insensitive)
        if self.source_dir.exists():
            debug("Iterating directory...")
            for f in self.source_dir.iterdir():
                if f.is_file() and f.suffix.lower() in self.extensions:
                    images.append(f)
            debug(f"Loop complete, found {len(images)}")

        debug(f"Found {len(images)} images")
        if images:
            debug(f"First few: {[img.name for img in images[:3]]}")

        return sorted(images)

    def _load_tags(self):
        """Load existing tags from file."""
        if self.output_file.exists():
            with open(self.output_file) as f:
                data = json.load(f)
                print(f"Loaded {len(data)} existing entries")
                return data
        return {}

    def _find_first_untagged(self):
        """Find index of first image without tags."""
        for i, img_path in enumerate(self.images):
            if img_path.name not in self.all_tags:
                return i
        # All tagged, return last image
        return max(0, len(self.images) - 1)

    def _jump_to_untagged(self):
        """Jump to next untagged image from current position."""
        # Look from current position forward
        for i in range(self.current_index + 1, len(self.images)):
            if self.images[i].name not in self.all_tags:
                self.current_index = i
                self._show_current_image()
                self.status_label.config(text=f"Jumped to untagged #{i+1}", fg='#00ffff')
                return

        # Wrap around to beginning
        for i in range(0, self.current_index):
            if self.images[i].name not in self.all_tags:
                self.current_index = i
                self._show_current_image()
                self.status_label.config(text=f"Jumped to untagged #{i+1} (wrapped)", fg='#00ffff')
                return

        self.status_label.config(text="All images are tagged!", fg='#00ff00')

    def _save_tags(self):
        """Save tags to file."""
        try:
            # Write to temp file first, then rename (atomic on most systems)
            temp_file = self.output_file.with_suffix('.json.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.all_tags, f, indent=2)

            # Replace original
            if self.output_file.exists():
                self.output_file.unlink()
            temp_file.rename(self.output_file)

            print(f"Saved {len(self.all_tags)} entries to {self.output_file}")
        except PermissionError as e:
            print(f"ERROR: Permission denied saving to {self.output_file}")
            print("Make sure the file isn't open in another program.")
            # Try backup location
            backup = Path.home() / 'tags_backup.json'
            try:
                with open(backup, 'w') as f:
                    json.dump(self.all_tags, f, indent=2)
                print(f"Saved backup to {backup}")
            except Exception:
                print("Could not save backup either!")
        except Exception as e:
            print(f"ERROR saving tags: {e}")

    def _setup_gui(self):
        """Setup the GUI layout."""
        self.root.configure(bg='#1a1a1a')

        # Main frame
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Image display
        self.image_label = tk.Label(main_frame, bg='#2a2a2a')
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Info frame
        info_frame = tk.Frame(main_frame, bg='#1a1a1a')
        info_frame.pack(fill=tk.X, pady=(10, 0))

        # Progress label
        self.progress_label = tk.Label(
            info_frame,
            text="0 / 0",
            font=('Consolas', 12),
            fg='white',
            bg='#1a1a1a'
        )
        self.progress_label.pack(side=tk.LEFT)

        # Filename label
        self.filename_label = tk.Label(
            info_frame,
            text="",
            font=('Consolas', 10),
            fg='#888888',
            bg='#1a1a1a'
        )
        self.filename_label.pack(side=tk.LEFT, padx=(20, 0))

        # Current tags display
        self.tags_label = tk.Label(
            main_frame,
            text="Tags: (none)",
            font=('Consolas', 14, 'bold'),
            fg='#00ffff',
            bg='#1a1a1a'
        )
        self.tags_label.pack(fill=tk.X, pady=(10, 0))

        # Status label
        self.status_label = tk.Label(
            main_frame,
            text="Toggle tags with keys, Enter to confirm",
            font=('Consolas', 11),
            fg='#888888',
            bg='#1a1a1a'
        )
        self.status_label.pack(fill=tk.X, pady=(5, 0))

        # Help frame
        help_frame = tk.Frame(main_frame, bg='#1a1a1a')
        help_frame.pack(fill=tk.X, pady=(10, 0))

        # Build help text
        help_lines = ["Tags (toggle on/off):"]
        for key, tag in sorted(self.tags.items()):
            help_lines.append(f"  [{key}] {tag}")
        help_lines.append("")
        help_lines.append("")
        help_lines.append("Controls: [Space/Enter] Next  [Backspace] Back  [U] Jump to untagged  [Esc] Quit")

        self.help_label = tk.Label(
            help_frame,
            text="\n".join(help_lines),
            font=('Consolas', 9),
            fg='#666666',
            bg='#1a1a1a',
            justify=tk.LEFT
        )
        self.help_label.pack(side=tk.LEFT)

        # Stats label
        self.stats_label = tk.Label(
            help_frame,
            text="",
            font=('Consolas', 9),
            fg='#666666',
            bg='#1a1a1a',
            justify=tk.RIGHT
        )
        self.stats_label.pack(side=tk.RIGHT)

        self.root.minsize(900, 700)

    def _bind_keys(self):
        """Bind keyboard events."""
        # Tag keys
        for key in self.tags.keys():
            self.root.bind(key.lower(), lambda e, k=key: self._toggle_tag(k))
            self.root.bind(key.upper(), lambda e, k=key: self._toggle_tag(k))

        # Control keys
        self.root.bind('<Return>', lambda e: self._confirm())
        self.root.bind('<space>', lambda e: self._skip())
        self.root.bind('<BackSpace>', lambda e: self._back())
        self.root.bind('<Escape>', lambda e: self._quit())
        self.root.bind('<Left>', lambda e: self._back())
        self.root.bind('<Right>', lambda e: self._confirm())

        # Navigation keys
        self.root.bind('u', lambda e: self._jump_to_untagged())
        self.root.bind('U', lambda e: self._jump_to_untagged())

    def _show_current_image(self):
        """Display the current image."""
        debug(f"_show_current_image: index={self.current_index}, total={len(self.images)}")
        if self.current_index >= len(self.images):
            debug("Index >= length, showing complete")
            self._show_complete()
            return

        image_path = self.images[self.current_index]

        # Load existing tags for this image
        key = str(image_path.name)
        if key in self.all_tags:
            self.current_tags = set(self.all_tags[key])
        else:
            self.current_tags = set()

        try:
            # Load and resize image
            img = Image.open(image_path)

            win_width = self.root.winfo_width()
            win_height = self.root.winfo_height()
            max_width = max(400, win_width - 40) if win_width > 100 else 860
            max_height = max(300, win_height - 280) if win_height > 300 else 500

            ratio = min(max_width / img.width, max_height / img.height)
            if ratio < 1:
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

            # Update labels
            self.progress_label.config(
                text=f"{self.current_index + 1} / {len(self.images)}"
            )
            self.filename_label.config(text=image_path.name)
            self._update_tags_display()
            self._update_stats_display()

        except Exception as e:
            debug(f"EXCEPTION loading image: {e}")
            import traceback
            debug(traceback.format_exc())
            self.status_label.config(text=f"Error loading image: {e}", fg='red')
            # Don't auto-advance on error - let user press space to skip
            # self.current_index += 1
            # self._show_current_image()

    def _toggle_tag(self, key):
        """Toggle a tag on/off for current image."""
        tag = self.tags[key]
        if tag in self.current_tags:
            self.current_tags.remove(tag)
        else:
            self.current_tags.add(tag)
        self._update_tags_display()

    def _update_tags_display(self):
        """Update the current tags display."""
        if self.current_tags:
            tags_str = ", ".join(sorted(self.current_tags))
            self.tags_label.config(text=f"Tags: {tags_str}", fg='#00ffff')
        else:
            self.tags_label.config(text="Tags: (none)", fg='#666666')

    def _confirm(self):
        """Confirm tags and move to next image."""
        if self.current_index >= len(self.images):
            return

        image_path = self.images[self.current_index]
        key = str(image_path.name)

        # Save tags
        if self.current_tags:
            self.all_tags[key] = list(self.current_tags)
        elif key in self.all_tags:
            del self.all_tags[key]

        # Auto-save to file
        self._save_tags()

        self.status_label.config(
            text=f"Saved: {', '.join(sorted(self.current_tags)) or '(cleared)'}",
            fg='#00ff00'
        )

        self.current_index += 1
        self._show_current_image()

    def _skip(self):
        """Skip if no tags, confirm if tags exist."""
        if self.current_tags:
            # Has tags - treat like confirm
            self._confirm()
        else:
            # No tags - skip
            self.current_index += 1
            self.status_label.config(text="Skipped", fg='#ffff00')
            self._show_current_image()

    def _back(self):
        """Go back to previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self._show_current_image()

    def _update_stats_display(self):
        """Update statistics display."""
        # Count tags
        tag_counts = {}
        for tags in self.all_tags.values():
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        lines = [f"Tagged: {len(self.all_tags)} / {len(self.images)}"]
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:8]:
            lines.append(f"  {tag}: {count}")

        self.stats_label.config(text="\n".join(lines))

    def _show_complete(self):
        """Show completion message."""
        self.image_label.config(image='')
        self.progress_label.config(text="Complete!")
        self.tags_label.config(text="All images processed!")
        self.status_label.config(text="Press Escape to save and quit", fg='#00ff00')
        self._update_stats_display()

    def _quit(self):
        """Save and quit."""
        self._save_tags()
        self.root.quit()
        self.root.destroy()

    def run(self):
        """Start the application."""
        print(f"Loaded {len(self.images)} images from {self.source_dir}")
        print(f"Output file: {self.output_file}")
        print(f"Previously tagged: {len(self.all_tags)}")
        print("\nStarting tagger...")
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description='Tag images with multiple labels')

    parser.add_argument('--source', '-s', type=str, required=True,
                       help='Source directory containing images')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output JSON file for tags')
    parser.add_argument('--tags', '-t', type=str, default=None,
                       help='JSON file with custom tags (key: tag_name)')
    parser.add_argument('--skip-tagged', '-u', action='store_true',
                       help='Start at first untagged image')
    parser.add_argument('--start', '-n', type=int, default=None,
                       help='Start at image number N (1-indexed)')

    args = parser.parse_args()

    # Load custom tags if provided
    tags = None
    if args.tags:
        with open(args.tags) as f:
            tags = json.load(f)

    # Run tagger
    tagger = ImageTagger(
        source_dir=args.source,
        output_file=args.output,
        tags=tags,
        skip_tagged=args.skip_tagged,
        start_index=args.start
    )
    tagger.run()


if __name__ == '__main__':
    main()
