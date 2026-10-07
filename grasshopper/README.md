# Menger sponge

A Grasshopper definition for a 3D Menger sponge. It uses two number sliders and one C# script component that ship with Rhino 8. No plug-ins are required.

## Open in Rhino

1. Start Rhino 8.
2. Run the `Grasshopper` command.
3. In Grasshopper, choose **File → Open** and select `menger-sponge.ghx` in this folder.
4. The sponge previews in the Rhino viewport, centered on the world origin.

## Sliders

- **Iterations** — whole numbers from 0 to 3. Default is 2. Zero is a solid cube. Each step removes the center and the face centers of every cube.
- **Size** — outer edge length. Default is 10.

## Bake

Right-click the **Mesh** output on the Menger component and choose **Bake**. That writes one mesh into the Rhino document.
