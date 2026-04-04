with open('index.html', 'r') as f:
    content = f.read()

# I need to sync index.html UI sliders with main.js globals properly. The problem was window.zpf_heat is not shared or index.html is entirely separate?
# Wait! index.html is the "Kinematic Weave" and main.js is the "Hydrogen Atom" WebGL FBO rendering.
# index.html uses calculatePhysicsBatch. main.js uses its own animate loop.
# Let me fix both.
