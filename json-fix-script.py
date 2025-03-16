#!/usr/bin/env python3
import json
import sys
import re


def fix_json_format(input_file, output_file):
    """
    Fix the JSON format of the ground markers file.

    This script specifically handles the RuneLite ground marker format by:
    1. Reading the input JSON file
    2. Fixing any escaped characters (like \: to :)
    3. Ensuring proper JSON formatting
    4. Writing the fixed JSON to the output file
    """
    try:
        # Read the input file
        with open(input_file, "r") as f:
            content = f.read()

        # If it's valid JSON already, just load and save to standardize
        try:
            data = json.loads(content)
            print(f"Input file is already valid JSON with {len(data)} entries")

            # Save in standardized format
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)

            print(f"Saved standardized JSON to {output_file}")
            return

        except json.JSONDecodeError:
            # Not valid JSON, need to fix it
            print("Input file is not valid JSON, attempting to fix...")

        # Fix common escaping issues in RuneLite ground marker JSON
        fixed_content = content

        # Replace escaped characters
        fixed_content = fixed_content.replace("\\:", ":")
        fixed_content = fixed_content.replace("\\,", ",")
        fixed_content = fixed_content.replace('\\"', '"')

        # Try to parse the fixed content
        try:
            data = json.loads(fixed_content)
            print(f"Fixed JSON successfully with {len(data)} entries")

            # Save the fixed JSON
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)

            print(f"Saved fixed JSON to {output_file}")
            return

        except json.JSONDecodeError as e:
            print(f"Simple fixes didn't work: {e}")

            # Try a more advanced approach - manual parsing
            # This assumes the content is a list of JSON objects
            objects = []

            # Extract each object using a regex pattern for JSON-like objects
            pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
            matches = re.findall(pattern, content)

            for match in matches:
                try:
                    # Fix escaping in each object
                    fixed_match = match.replace("\\:", ":")
                    fixed_match = fixed_match.replace("\\,", ",")
                    fixed_match = fixed_match.replace('\\"', '"')

                    # Parse the object
                    obj = json.loads(fixed_match)
                    objects.append(obj)
                except json.JSONDecodeError:
                    print(f"Could not parse object: {match[:50]}...")

            if objects:
                print(f"Extracted {len(objects)} valid objects manually")

                # Save the extracted objects
                with open(output_file, "w") as f:
                    json.dump(objects, f, indent=2)

                print(f"Saved extracted objects to {output_file}")
                return

            # If all else fails, try a very permissive line-by-line approach
            print("Attempting line-by-line parsing...")
            objects = []
            lines = content.split("\n")

            for line in lines:
                line = line.strip()
                if not line or line in "[]{}," or line.startswith("//"):
                    continue

                # Look for JSON-like objects in the line
                pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
                matches = re.findall(pattern, line)

                for match in matches:
                    try:
                        # Fix escaping in each object
                        fixed_match = match.replace("\\:", ":")
                        fixed_match = fixed_match.replace("\\,", ",")
                        fixed_match = fixed_match.replace('\\"', '"')

                        # Parse the object
                        obj = json.loads(fixed_match)
                        objects.append(obj)
                    except json.JSONDecodeError:
                        continue

            if objects:
                print(f"Extracted {len(objects)} objects with line-by-line parsing")

                # Save the extracted objects
                with open(output_file, "w") as f:
                    json.dump(objects, f, indent=2)

                print(f"Saved objects to {output_file}")
                return

            print("Failed to fix the JSON format. Manual intervention required.")
            return

    except Exception as e:
        print(f"Error processing file: {e}")
        return


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fix_json.py input_file.json output_file.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    fix_json_format(input_file, output_file)
