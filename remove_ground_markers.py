#!/usr/bin/env python3
import json
import os
import sys
import re
from pathlib import Path
import argparse

# For Flatpak Jagex Launcher installation with profiles
DEFAULT_PROFILES_PATH = (
    Path.home() / ".var" / "app" / "com.jagex.Launcher" / ".runelite" / "profiles2"
)


def load_markers_to_remove_from_json(json_file):
    """Load markers to remove from a JSON file."""
    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            print(f"Loaded {len(data)} markers to remove from JSON file")
            return data
        else:
            print("Error: Unexpected JSON format. Expected a list of marker objects.")
            return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {json_file}")
        return []
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return []


def group_markers_by_region(markers):
    """Group markers by their region ID."""
    grouped = {}
    for marker in markers:
        region_id = str(marker.get("regionId"))
        if region_id:
            if region_id not in grouped:
                grouped[region_id] = []
            grouped[region_id].append(marker)

    return grouped


def create_marker_patterns(markers):
    """Create regex patterns to match markers in the file."""
    patterns = []

    for marker in markers:
        # Get the key attributes for matching
        region_id = marker.get("regionId")
        region_x = marker.get("regionX")
        region_y = marker.get("regionY")
        z = marker.get("z")

        if all(v is not None for v in [region_id, region_x, region_y, z]):
            # Create a pattern that matches this marker
            # This is a simplified pattern that looks for the key coordinates
            pattern = rf'({{"regionId"[^,]*{region_id}[^,]*,"regionX"[^,]*{region_x}[^,]*,"regionY"[^,]*{region_y}[^,]*,"z"[^,]*{z}[^}}]*}})'
            # Also try the variant with escaped quotes
            pattern2 = rf'(\\{{"regionId"[^,]*{region_id}[^,]*,"regionX"[^,]*{region_x}[^,]*,"regionY"[^,]*{region_y}[^,]*,"z"[^,]*{z}[^}}]*\\}})'
            patterns.append(pattern)
            patterns.append(pattern2)

    return patterns


def process_profile_file(profile_file, markers_to_remove, dry_run=False):
    """Process a profile file to remove specific ground markers."""
    print(f"\nProcessing profile: {os.path.basename(profile_file)}")

    # Group markers by region for easier processing
    markers_by_region = group_markers_by_region(markers_to_remove)

    # Read the profile file
    try:
        with open(profile_file, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading profile file: {e}")
        return

    # Create a backup of the profile file
    if not dry_run:
        backup_file = f"{profile_file}.backup"
        with open(backup_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created backup at {backup_file}")

    # Find all region marker entries
    region_entries = re.findall(r"groundMarker\.region_(\d+)=(.+)(?:\n|$)", content)

    if not region_entries:
        print("No ground markers found in profile")
        return

    print(f"Found {len(region_entries)} region marker entries")

    # Process each region entry
    modified_content = content
    total_markers_removed = 0

    for region_id, entry_data in region_entries:
        if region_id not in markers_by_region:
            continue

        # Get markers to remove for this region
        region_markers_to_remove = markers_by_region[region_id]
        print(
            f"Processing region {region_id} with {len(region_markers_to_remove)} markers to remove"
        )

        # Create patterns to match these markers
        patterns = create_marker_patterns(region_markers_to_remove)

        # Original region entry
        original_entry = f"groundMarker.region_{region_id}={entry_data}"
        modified_entry = original_entry

        # Count markers in the region
        marker_count = modified_entry.count("regionId")

        # Apply each pattern to remove markers
        markers_removed = 0
        for pattern in patterns:
            # Check if pattern matches anything
            if re.search(pattern, modified_entry):
                # Remove the marker, paying attention to commas
                def replacement(match):
                    # If we're not in dry run mode, count the removal
                    nonlocal markers_removed
                    markers_removed += 1
                    # Return empty string to remove the marker
                    return ""

                modified_entry = re.sub(pattern + r",?", replacement, modified_entry)
                # Also handle the case where it's the last item with no comma
                modified_entry = re.sub(r",?" + pattern, replacement, modified_entry)

        # Clean up any potential double commas left behind
        modified_entry = re.sub(r",\s*,", ",", modified_entry)
        # Clean up empty brackets
        modified_entry = re.sub(r"\[\s*,", "[", modified_entry)
        modified_entry = re.sub(r",\s*\]", "]", modified_entry)
        modified_entry = re.sub(r"\[\s*\]", "[]", modified_entry)

        # Update the total count
        total_markers_removed += markers_removed

        if markers_removed > 0:
            print(f"Removed {markers_removed} markers from region {region_id}")

            # If all markers were removed, remove the entire entry
            if modified_entry.count("regionId") == 0 or "[]" in modified_entry:
                print(f"All markers removed from region {region_id}, removing entry")
                modified_content = modified_content.replace(original_entry, "")
            else:
                # Otherwise, update the entry
                modified_content = modified_content.replace(
                    original_entry, modified_entry
                )

    # Write the modified content back to the file
    if not dry_run and total_markers_removed > 0:
        # Clean up any blank lines
        modified_content = re.sub(r"\n\s*\n", "\n", modified_content)

        with open(profile_file, "w", encoding="utf-8") as f:
            f.write(modified_content)

        print(f"Updated profile file with {total_markers_removed} markers removed")
    elif dry_run and total_markers_removed > 0:
        print(f"Dry run - would have removed {total_markers_removed} markers")
    else:
        print("No markers were removed")


def main():
    parser = argparse.ArgumentParser(
        description="Remove specific ground markers directly from RuneLite profiles"
    )
    parser.add_argument(
        "json_file", help="JSON file containing ground markers to remove"
    )
    parser.add_argument("--profiles-dir", help="Path to RuneLite profiles directory")
    parser.add_argument("--profile", help="Specific profile file to modify")
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not actually modify the markers"
    )

    args = parser.parse_args()

    # Find RuneLite profiles directory
    if args.profiles_dir:
        profiles_path = Path(args.profiles_dir)
    else:
        profiles_path = DEFAULT_PROFILES_PATH

    print(f"Using RuneLite profiles directory: {profiles_path}")

    # Load markers to remove
    markers_to_remove = load_markers_to_remove_from_json(args.json_file)
    if not markers_to_remove:
        print("No markers to remove. Exiting.")
        return

    # Find profile files
    if args.profile:
        profile_file = profiles_path / args.profile
        if not profile_file.exists():
            print(f"Error: Profile file not found: {args.profile}")
            return
        profile_files = [profile_file]
    else:
        profile_files = list(profiles_path.glob("*.properties"))

    if not profile_files:
        print("No profile files found")
        return

    # Process each profile
    for profile_file in profile_files:
        process_profile_file(profile_file, markers_to_remove, args.dry_run)


if __name__ == "__main__":
    main()
