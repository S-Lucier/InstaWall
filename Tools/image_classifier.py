"""
Image Classifier Tool

A simple GUI tool for quickly categorizing images into folders.
Shows each image and moves it to a folder based on which key you press.

Usage:
    python image_classifier.py --source ./unsorted_images --output ./sorted

Controls:
    1-9, a-z: Move to corresponding category folder
    Space: Skip (leave in place)
    Backspace: Undo last action
    Escape: Quit
    Delete: Move to 'trash' folder
"""

import argparse
import json
import shutil
import sys
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk


class ImageClassifier:
    """GUI application for classifying images into folders."""

    def __init__(self, source_dir, output_dir, categories=None, extensions=None):
        """
        Args:
            source_dir: Directory containing images to classify
            output_dir: Directory where category folders will be created
            categories: Dict mapping keys to category names
                       If None, uses default categories
            extensions: Valid image extensions
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.extensions = extensions or {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}

        # Default categories - customize as needed
        self.categories = categories or {
            '1': 'dungeon',
            '2': 'outdoor',
            '3': 'urban',
            '4': 'cave',
            '5': 'ship',
            '6': 'other',
            'd': 'discard',
            'g': 'has_grid',
            'n': 'no_grid',
            'h': 'high_quality',
            'l': 'low_quality',
        }

        # Create category folders
        for category in self.categories.values():
            (self.output_dir / category).mkdir(exist_ok=True)

        # Trash folder for deleted images
        (self.output_dir / '_trash').mkdir(exist_ok=True)
        (self.output_dir / '_skipped').mkdir(exist_ok=True)

        # Collect images
        self.images = self._collect_images()
        self.current_index = 0
        self.history = []  # For undo

        # Stats
        self.stats = {cat: 0 for cat in self.categories.values()}
        self.stats['_skipped'] = 0
        self.stats['_trash'] = 0

        # Setup GUI
        self.root = tk.Tk()
        self.root.title("Image Classifier")
        self._setup_gui()

        # Bind keys
        self._bind_keys()

        # Load first image
        if self.images:
            self._show_current_image()
        else:
            self.status_label.config(text="No images found!")

    def _collect_images(self):
        """Collect all image files from source directory."""
        images = []
        for ext in self.extensions:
            images.extend(self.source_dir.glob(f'*{ext}'))
            images.extend(self.source_dir.glob(f'*{ext.upper()}'))
        return sorted(set(images))

    def _setup_gui(self):
        """Setup the GUI layout."""
        # Configure window
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

        # Status label
        self.status_label = tk.Label(
            main_frame,
            text="",
            font=('Consolas', 11),
            fg='#00ff00',
            bg='#1a1a1a'
        )
        self.status_label.pack(fill=tk.X, pady=(5, 0))

        # Help frame
        help_frame = tk.Frame(main_frame, bg='#1a1a1a')
        help_frame.pack(fill=tk.X, pady=(10, 0))

        # Build help text
        help_lines = ["Categories:"]
        for key, category in sorted(self.categories.items()):
            help_lines.append(f"  [{key}] {category}")
        help_lines.append("")
        help_lines.append("Controls: [Space] Skip  [Backspace] Undo  [Delete] Trash  [Esc] Quit")

        help_text = "\n".join(help_lines)
        self.help_label = tk.Label(
            help_frame,
            text=help_text,
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

        # Set minimum window size
        self.root.minsize(800, 600)

    def _bind_keys(self):
        """Bind keyboard events."""
        # Category keys
        for key in self.categories.keys():
            self.root.bind(key.lower(), lambda e, k=key: self._classify(k))
            self.root.bind(key.upper(), lambda e, k=key: self._classify(k))

        # Control keys
        self.root.bind('<space>', lambda e: self._skip())
        self.root.bind('<BackSpace>', lambda e: self._undo())
        self.root.bind('<Delete>', lambda e: self._trash())
        self.root.bind('<Escape>', lambda e: self._quit())

        # Arrow keys for navigation without moving
        self.root.bind('<Left>', lambda e: self._navigate(-1))
        self.root.bind('<Right>', lambda e: self._navigate(1))

    def _show_current_image(self):
        """Display the current image."""
        if self.current_index >= len(self.images):
            self._show_complete()
            return

        image_path = self.images[self.current_index]

        try:
            # Load and resize image
            img = Image.open(image_path)

            # Calculate display size (fit within window)
            max_width = self.root.winfo_width() - 40 or 760
            max_height = self.root.winfo_height() - 200 or 400

            # Maintain aspect ratio
            ratio = min(max_width / img.width, max_height / img.height)
            if ratio < 1:
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep reference

            # Update labels
            self.progress_label.config(
                text=f"{self.current_index + 1} / {len(self.images)}"
            )
            self.filename_label.config(text=image_path.name)
            self.status_label.config(text="", fg='#00ff00')

            # Update stats
            self._update_stats_display()

        except Exception as e:
            self.status_label.config(text=f"Error loading image: {e}", fg='red')
            self._next()

    def _classify(self, key):
        """Move current image to category folder."""
        if self.current_index >= len(self.images):
            return

        category = self.categories[key]
        image_path = self.images[self.current_index]
        dest_path = self.output_dir / category / image_path.name

        # Handle duplicate names
        if dest_path.exists():
            stem = dest_path.stem
            suffix = dest_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = self.output_dir / category / f"{stem}_{counter}{suffix}"
                counter += 1

        # Move file
        shutil.move(str(image_path), str(dest_path))

        # Record for undo
        self.history.append({
            'action': 'classify',
            'from': image_path,
            'to': dest_path,
            'index': self.current_index
        })

        # Update stats
        self.stats[category] += 1

        # Show status
        self.status_label.config(text=f"→ {category}", fg='#00ff00')

        # Remove from list and show next
        self.images.pop(self.current_index)
        self._show_current_image()

    def _skip(self):
        """Skip current image without moving it."""
        if self.current_index >= len(self.images):
            return

        self.stats['_skipped'] += 1
        self.status_label.config(text="Skipped", fg='#ffff00')
        self.current_index += 1
        self._show_current_image()

    def _trash(self):
        """Move current image to trash folder."""
        if self.current_index >= len(self.images):
            return

        image_path = self.images[self.current_index]
        dest_path = self.output_dir / '_trash' / image_path.name

        # Handle duplicates
        if dest_path.exists():
            stem = dest_path.stem
            suffix = dest_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = self.output_dir / '_trash' / f"{stem}_{counter}{suffix}"
                counter += 1

        shutil.move(str(image_path), str(dest_path))

        self.history.append({
            'action': 'trash',
            'from': image_path,
            'to': dest_path,
            'index': self.current_index
        })

        self.stats['_trash'] += 1
        self.status_label.config(text="→ trash", fg='#ff6666')

        self.images.pop(self.current_index)
        self._show_current_image()

    def _undo(self):
        """Undo last action."""
        if not self.history:
            self.status_label.config(text="Nothing to undo", fg='#ff6666')
            return

        last = self.history.pop()

        # Move file back
        shutil.move(str(last['to']), str(last['from']))

        # Restore to list
        self.images.insert(last['index'], last['from'])
        self.current_index = last['index']

        # Update stats
        if last['action'] == 'classify':
            category = last['to'].parent.name
            self.stats[category] -= 1
        elif last['action'] == 'trash':
            self.stats['_trash'] -= 1

        self.status_label.config(text="Undone", fg='#ffff00')
        self._show_current_image()

    def _navigate(self, direction):
        """Navigate without classifying (for preview)."""
        new_index = self.current_index + direction
        if 0 <= new_index < len(self.images):
            self.current_index = new_index
            self._show_current_image()

    def _next(self):
        """Move to next image."""
        self.current_index += 1
        self._show_current_image()

    def _update_stats_display(self):
        """Update the stats display."""
        lines = ["Stats:"]
        for category, count in sorted(self.stats.items()):
            if count > 0:
                lines.append(f"  {category}: {count}")
        self.stats_label.config(text="\n".join(lines))

    def _show_complete(self):
        """Show completion message."""
        self.image_label.config(image='')
        self.progress_label.config(text="Complete!")
        self.filename_label.config(text="")
        self.status_label.config(
            text="All images classified! Press Escape to quit.",
            fg='#00ff00'
        )
        self._update_stats_display()
        self._save_stats()

    def _save_stats(self):
        """Save classification stats to file."""
        stats_path = self.output_dir / 'classification_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"Stats saved to {stats_path}")

    def _quit(self):
        """Quit the application."""
        self._save_stats()
        print("\nFinal stats:")
        for category, count in sorted(self.stats.items()):
            if count > 0:
                print(f"  {category}: {count}")
        print(f"\nRemaining: {len(self.images)} images")
        self.root.quit()
        self.root.destroy()

    def run(self):
        """Start the application."""
        print(f"Loaded {len(self.images)} images from {self.source_dir}")
        print(f"Output directory: {self.output_dir}")
        print("\nCategories:")
        for key, category in sorted(self.categories.items()):
            print(f"  [{key}] {category}")
        print("\nStarting classifier...")
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description='Classify images into folders')

    parser.add_argument('--source', '-s', type=str, required=True,
                       help='Source directory containing images')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for sorted images')
    parser.add_argument('--categories', '-c', type=str, default=None,
                       help='JSON file with custom categories (key: folder_name)')

    args = parser.parse_args()

    # Load custom categories if provided
    categories = None
    if args.categories:
        with open(args.categories) as f:
            categories = json.load(f)

    # Run classifier
    classifier = ImageClassifier(
        source_dir=args.source,
        output_dir=args.output,
        categories=categories
    )
    classifier.run()


if __name__ == '__main__':
    main()
