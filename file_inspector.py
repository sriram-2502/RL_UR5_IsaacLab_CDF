#!/usr/bin/env python3

"""Simple script to inspect USD file structure using basic Isaac Sim methods."""

import sys
import os
import argparse
import traceback

def inspect_usd_file(usd_path):
    """Use basic Isaac Sim methods to inspect USD file."""
    print(f"\n=== INSPECTING USD FILE: {usd_path} ===")
    
    # Import Isaac Sim modules
    try:
        import omni.usd
        import omni.kit
        from omni.isaac.kit import SimulationApp
    except ImportError:
        print("Error: This script must be run within Isaac Sim environment")
        return
    
    # Initialize simulation app in headless mode for USD inspection
    sim_app = SimulationApp({"headless": True})
    
    try:
        # Load the USD file
        usd_context = omni.usd.get_context()
        usd_context.open_stage(usd_path, None)
        stage = usd_context.get_stage()
        
        if not stage:
            print(f"Failed to open USD stage at {usd_path}")
            return
        
        print(f"Successfully loaded USD file: {usd_path}")
        
        # Simple function to traverse prims by path
        def print_prims_at_path(base_path, indent=0):
            import omni.usd
            
            indent_str = "  " * indent
            print(f"{indent_str}Path: {base_path}")
            
            # Get prim at path
            prim = stage.GetPrimAtPath(base_path)
            if not prim:
                print(f"{indent_str}  No prim at path")
                return
            
            # Print basic info
            print(f"{indent_str}  Type: {prim.GetTypeName()}")
            print(f"{indent_str}  Is Valid: {prim.IsValid()}")
            
            # Try to get schemas
            try:
                print(f"{indent_str}  Applied Schemas:")
                for schema in prim.GetAppliedSchemas():
                    print(f"{indent_str}    {schema}")
                    # Check for articulation API
                    if "ArticulationRoot" in schema:
                        print(f"{indent_str}    *** FOUND ARTICULATION ROOT API ***")
            except Exception as e:
                print(f"{indent_str}  Error getting schemas: {e}")
            
            # Check property names
            try:
                prop_names = prim.GetPropertyNames()
                if prop_names:
                    print(f"{indent_str}  Properties: {', '.join(prop_names[:10])}...")
                    if "joint" in base_path.lower() or "link" in base_path.lower():
                        print(f"{indent_str}  All properties: {', '.join(prop_names)}")
            except Exception as e:
                print(f"{indent_str}  Error getting properties: {e}")
            
            # Get child paths
            try:
                # First try GetChildren
                children = []
                try:
                    children = prim.GetChildren()
                except:
                    # If GetChildren fails, try to get children using GetPaths
                    children_paths = []
                    # List subdirectories manually by appending path
                    for name in prim.GetNames():
                        if base_path == "/":
                            child_path = f"/{name}"
                        else:
                            child_path = f"{base_path}/{name}"
                        children_paths.append(child_path)
                    
                    # Get prims from paths
                    children = [stage.GetPrimAtPath(path) for path in children_paths]
                
                # Print child info
                if children:
                    print(f"{indent_str}  Children: {len(children)}")
                    for child in children:
                        try:
                            child_path = child.GetPath().pathString
                        except:
                            child_path = str(child)
                        
                        print_prims_at_path(child_path, indent + 1)
                else:
                    print(f"{indent_str}  No children")
            
            except Exception as e:
                print(f"{indent_str}  Error getting children: {e}")
        
        # Start traversal from root
        print("\n=== ROOT LEVEL PRIMS ===")
        try:
            # Try to get the root path
            root_path = "/"
            print_prims_at_path(root_path)
        except Exception as e:
            print(f"Error traversing from root: {e}")
            traceback.print_exc()
        
        # Look for specific important paths
        print("\n=== SEARCHING FOR KEY PATHS ===")
        key_paths = [
            "/ur5_robot",
            "/robot",
            "/Robot",
            "/ur5",
            "/UR5"
        ]
        
        for path in key_paths:
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                print(f"Found prim at {path}")
                print(f"  Type: {prim.GetTypeName()}")
                try:
                    print(f"  Schemas: {list(prim.GetAppliedSchemas())}")
                except:
                    print("  Could not get schemas")
                
                # Check for articulation API
                try:
                    has_articulation = False
                    for schema in prim.GetAppliedSchemas():
                        if "ArticulationRoot" in schema:
                            has_articulation = True
                            break
                    print(f"  Has ArticulationRootAPI: {has_articulation}")
                except:
                    print("  Could not check for articulation API")
        
        print("\n=== SUMMARY AND RECOMMENDATIONS ===")
        print("* If no articulation root was found, you need to apply it to the root prim")
        print("* Check for exact paths of base_link and ee_link")
        print("* Make sure your config uses the correct prim paths")
        
    except Exception as e:
        print(f"Error during USD inspection: {e}")
        traceback.print_exc()
    finally:
        # Close the simulation app
        sim_app.close()
        print("=== USD INSPECTION COMPLETE ===\n")

def main():
    parser = argparse.ArgumentParser(description="Simple USD file inspector for Isaac Sim")
    parser.add_argument("usd_path", help="Path to the USD file to inspect")
    args = parser.parse_args()
    
    if not os.path.exists(args.usd_path):
        print(f"Error: USD file not found at {args.usd_path}")
        return 1
    
    inspect_usd_file(args.usd_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())