// Topological Geon Physics Engine

// Initialize Three.js scene, camera, and renderer
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 50;

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.autoClearColor = false; // We will clear manually when needed for the FBO blur
document.body.appendChild(renderer.domElement);

// -----------------------------------------------------------------------------
// Framebuffer Object (FBO) Accumulation (Time-Lapse Optical Blur)
// -----------------------------------------------------------------------------
// To simulate the "probability cloud" deterministically, we render the scene into an FBO,
// fade it slightly (e.g. 95% opacity), and draw it back on top of itself.
// This visually smears the high-speed localized particles into a continuous 3D orbital shell.

// Create two render targets for ping-ponging the feedback loop
let renderTarget1 = new THREE.WebGLRenderTarget(window.innerWidth, window.innerHeight, { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, format: THREE.RGBAFormat });
let renderTarget2 = new THREE.WebGLRenderTarget(window.innerWidth, window.innerHeight, { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, format: THREE.RGBAFormat });

// A separate scene/camera just to draw the faded previous frame onto the screen
const fboScene = new THREE.Scene();
const fboCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const fboMaterial = new THREE.ShaderMaterial({
    uniforms: {
        tDiffuse: { value: null },
        opacity: { value: 0.95 } // Smear persistence factor
    },
    vertexShader: `
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = vec4(position, 1.0);
        }
    `,
    fragmentShader: `
        uniform sampler2D tDiffuse;
        uniform float opacity;
        varying vec2 vUv;
        void main() {
            vec4 texColor = texture2D(tDiffuse, vUv);
            // We gently darken the rgb along with alpha so the FBO trail fades to black
            // rather than building up an infinitely bright additive white cloud.
            gl_FragColor = vec4(texColor.rgb * opacity, texColor.a * opacity);
        }
    `,
    transparent: true,
    blending: THREE.NormalBlending,
    depthWrite: false
});

const fboQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), fboMaterial);
fboScene.add(fboQuad);

// -----------------------------------------------------------------------------
// Core Physics Constants & Vacuum Fluid Material
// -----------------------------------------------------------------------------

// Planck's constant as kinematic action viscosity (arbitrary scale for viz)
const PLANCK_H = 6.626e-34;
const VISCOSITY_SCALE = 1.0;

// Speed of light strictly internal wave velocity
const SPEED_OF_LIGHT = 299792458; // c

const vacuumFluidUniforms = {
    uTime: { value: 0.0 },
    uPlanckH: { value: PLANCK_H * VISCOSITY_SCALE },
    uSpeedOfLight: { value: SPEED_OF_LIGHT }
};

const vacuumFluidVertexShader = `
    varying vec2 vUv;
    varying vec3 vPosition;
    varying vec3 vNormal;

    void main() {
        vUv = uv;
        vPosition = position;
        vNormal = normal;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

const vacuumFluidFragmentShader = `
    uniform float uTime;
    uniform float uPlanckH;
    uniform float uSpeedOfLight;

    varying vec2 vUv;
    varying vec3 vPosition;
    varying vec3 vNormal;

    void main() {
        // Zero-point tensor fluid active fluctuations
        // Simulated via simple sine waves based on time and position for visualization
        float fluctuation = sin(vPosition.x * 10.0 + uTime * 5.0) *
                            cos(vPosition.y * 10.0 + uTime * 5.0) * 0.5 + 0.5;

        // Base color representing vacuum energy density
        vec3 vacuumColor = vec3(0.05, 0.1, 0.2);
        vec3 activeColor = vec3(0.1, 0.3, 0.5) * fluctuation;

        gl_FragColor = vec4(vacuumColor + activeColor, 0.8);
    }
`;

const vacuumMaterial = new THREE.ShaderMaterial({
    uniforms: vacuumFluidUniforms,
    vertexShader: vacuumFluidVertexShader,
    fragmentShader: vacuumFluidFragmentShader,
    transparent: true,
    side: THREE.DoubleSide
});

// Basic light setup
const ambientLight = new THREE.AmbientLight(0x404040); // Soft white light
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
directionalLight.position.set(10, 10, 10).normalize();
scene.add(directionalLight);

// -----------------------------------------------------------------------------
// Because scene.background completely erases the previous render buffer when draw calls happen,
// we DO NOT set a background. The buffer naturally initializes clear (or black depending on
// context) and we explicitly handle clearing.
// scene.background = null; (Default)

// -----------------------------------------------------------------------------
// Particle Topologies (Geons)
// -----------------------------------------------------------------------------
// Parametric Geometry functions to restore the strict continuous box-tubes

function createMobiusDoubleLoopGeometry(radius, tube, radialSegments, tubularSegments) {
    const geometry = new THREE.BufferGeometry();
    const vertices = [];
    const indices = [];
    const uvs = [];

    for (let i = 0; i <= tubularSegments; i++) {
        // u goes from 0 to 4π for the double loop path around the major radius
        const u = (i / tubularSegments) * 4 * Math.PI;

        for (let j = 0; j <= radialSegments; j++) {
            // v goes from 0 to 2π for the fully closed cross-sectional tube
            const v = (j / radialSegments) * 2 * Math.PI;

            // Mobius strip parameterized as a 3D tube:
            // To create the twist, the cross-section rotates by u/2 as it travels along u.
            const twist = u / 2;

            // Calculate cross-section coordinates local to the curve
            const cx = tube * Math.cos(v);
            const cy = tube * Math.sin(v);

            // Apply the Möbius twist to the cross-section
            const twistedX = cx * Math.cos(twist) - cy * Math.sin(twist);
            const twistedY = cx * Math.sin(twist) + cy * Math.cos(twist);

            // Map the twisted cross-section onto the major circular path (radius)
            const x = (radius + twistedX) * Math.cos(u / 2); // Map 4pi u back to 2pi circular path mapping
            const y = (radius + twistedX) * Math.sin(u / 2);
            const z = twistedY;

            vertices.push(x, y, z);

            // Output UVs so shaders can map gradients along the 4pi tube
            uvs.push(i / tubularSegments, j / radialSegments);
        }
    }

    // Generate indices (Watertight mesh)
    for (let i = 0; i < tubularSegments; i++) {
        for (let j = 0; j < radialSegments; j++) {
            // Since we loop `i <= tubularSegments` and `j <= radialSegments`, the arrays have
            // (tubularSegments + 1) * (radialSegments + 1) vertices. This creates overlapping
            // vertices at the seam. We connect them into solid triangles.
            const a = i * (radialSegments + 1) + j;
            const b = (i + 1) * (radialSegments + 1) + j;
            const c = (i + 1) * (radialSegments + 1) + (j + 1);
            const d = i * (radialSegments + 1) + (j + 1);

            // Two triangles per face (making a quad)
            indices.push(a, b, d);
            indices.push(b, c, d);
        }
    }

    geometry.setIndex(indices);
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geometry.computeVertexNormals();
    return geometry;
}

function createTrefoilKnotGeometry(radius, tube, tubularSegments, radialSegments, p, q) {
    // A standard TorusKnotGeometry works well for (3,2) trefoil knots (proton)
    // It automatically generates UVs natively.
    return new THREE.TorusKnotGeometry(radius, tube, tubularSegments, radialSegments, p, q);
}

// -----------------------------------------------------------------------------
// Visual Representation (Light Packets in Casimir Fluid)
// -----------------------------------------------------------------------------
// Restore custom fluid-packet shaders. The continuous geometries are rendered
// using UV coordinates to animate glowing "light packets" oscillating circularly
// inside the knot tubes.
const lightPacketVertexShader = `
    varying vec2 vUv;
    varying vec3 vNormal;
    void main() {
        vUv = uv;
        vNormal = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

const lightPacketFragmentShader = `
    uniform vec3 color;
    uniform float time;
    uniform float speed;
    uniform float packetDensity;

    varying vec2 vUv;
    varying vec3 vNormal;

    void main() {
        // Create repeating light "packets" traveling circularly along the u-axis
        float wave = sin(vUv.x * packetDensity - time * speed) * 0.5 + 0.5;

        // Base edge glow
        float intensity = pow(0.7 - dot(vNormal, vec3(0, 0, 1.0)), 2.0);

        // Combine wave packets with the glow
        float finalAlpha = wave * intensity * 3.0;
        vec3 finalColor = color * wave + (color * 0.3);

        gl_FragColor = vec4(finalColor, finalAlpha);
    }
`;

// In WebGL, when drawing custom Parametric Geometries (like our Möbius loop) with `gl.TRIANGLES`,
// backface culling issues or depth sorting issues can make the continuous tube look segmented or broken
// when using pure AdditiveBlending across overlapping geometry.
// We use NormalBlending + precise depth testing to make it visually "watertight".

const electronMaterial = new THREE.ShaderMaterial({
    uniforms: {
        color: { value: new THREE.Color(0x00ffff) },
        time: { value: 0.0 },
        speed: { value: 15.0 },
        packetDensity: { value: 60.0 }
    },
    vertexShader: lightPacketVertexShader,
    fragmentShader: lightPacketFragmentShader,
    transparent: true,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide, // Essential for Moebius strips
    depthTest: true,
    depthWrite: false // Allow particles inside the tube to overlap without z-fighting
});

const protonMaterial = new THREE.ShaderMaterial({
    uniforms: {
        color: { value: new THREE.Color(0xff00ff) },
        time: { value: 0.0 },
        speed: { value: 3.0 },      // Slower macroscopic internal rotation proxy for heavy mass
        packetDensity: { value: 150.0 } // Highly compressed dense knot
    },
    vertexShader: lightPacketVertexShader,
    fragmentShader: lightPacketFragmentShader,
    transparent: true,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    depthWrite: false
});

// -----------------------------------------------------------------------------
// Hydrogen Atom Model (Topological Geon Framework)
// -----------------------------------------------------------------------------
const PROTON_MASS_PROXY = 50.0;
const ELECTRON_MASS_PROXY = 1.0;

const BOHR_RADIUS_PROXY = 40.0;
const COULOMB_FORCE_CONSTANT = 400.0;
const GRAVITY_CONSTANT = 0.5;

// Common Barycenter tracking
const protonRadius = BOHR_RADIUS_PROXY * (ELECTRON_MASS_PROXY / (PROTON_MASS_PROXY + ELECTRON_MASS_PROXY));
const electronRadius = BOHR_RADIUS_PROXY * (PROTON_MASS_PROXY / (PROTON_MASS_PROXY + ELECTRON_MASS_PROXY));

// The Proton: A highly compressed, dense, tiny (3,2)-trefoil knot
// Relative scale: The proton's radius is ~0.84 fm.
const protonGeometry = createTrefoilKnotGeometry(0.5, 0.15, 256, 32, 3, 2);
const protonMesh = new THREE.Mesh(protonGeometry, protonMaterial);
protonMesh.position.set(-protonRadius, 0, 0);
scene.add(protonMesh);

// The Electron: A loose, large, extended 4pi Möbius double-loop
// Relative scale: The electron's track/Compton radius is macroscopic compared to the proton (~386 fm).
const electronGeometry = createMobiusDoubleLoopGeometry(3.5, 0.4, 64, 512); // Higher resolution to ensure smooth loop
const electronMesh = new THREE.Mesh(electronGeometry, electronMaterial);
electronMesh.position.set(electronRadius, 0, 0);
scene.add(electronMesh);

// Required orbital velocity for a circular orbit around the barycenter
// v = sqrt(F_inward * r_from_barycenter / m)
const forceAtDistance = COULOMB_FORCE_CONSTANT / Math.pow(BOHR_RADIUS_PROXY, 2);
const electronSpeed = Math.sqrt(forceAtDistance * electronRadius / ELECTRON_MASS_PROXY);
const protonSpeed = Math.sqrt(forceAtDistance * protonRadius / PROTON_MASS_PROXY);

const electronVelocity = new THREE.Vector3(0, electronSpeed, 0);
const protonVelocity = new THREE.Vector3(0, -protonSpeed, 0);


// -----------------------------------------------------------------------------
// Arrow Helpers for Visualizations
// -----------------------------------------------------------------------------
function createRadialVectors(mesh, color, count, directionSign, length) {
    const helpers = [];
    for (let i = 0; i < count; i++) {
        const dir = new THREE.Vector3(
            Math.cos(i / count * Math.PI * 2),
            Math.sin(i / count * Math.PI * 2),
            0
        ).normalize().multiplyScalar(directionSign);
        const origin = new THREE.Vector3(0, 0, 0); // Local to mesh
        // Thin out the arrows for a cleaner aesthetic
        const arrow = new THREE.ArrowHelper(dir, origin, length, color, 0.3, 0.3);
        helpers.push(arrow);
        mesh.add(arrow);
    }
    return helpers;
}

// Removing rigid-body `ArrowHelpers` as the particles are no longer rigid meshes,
// they are dynamic emergent waves.

// Global Arrows for Coulomb interaction (Gauss shell projection)
const coulombArrowElectron = new THREE.ArrowHelper(new THREE.Vector3(1,0,0), electronMesh.position, 1, 0x00ffff, 1, 1);
const coulombArrowProton = new THREE.ArrowHelper(new THREE.Vector3(-1,0,0), protonMesh.position, 1, 0x00ffff, 1, 1);
scene.add(coulombArrowElectron);
scene.add(coulombArrowProton);


// DOM Elements for UI
const valH = document.getElementById('val-h');
const valC = document.getElementById('val-c');
const valVel = document.getElementById('val-vel');
const valDist = document.getElementById('val-dist');
const valCasimir = document.getElementById('val-casimir');
const valShadow = document.getElementById('val-shadow');
const valCentrifugal = document.getElementById('val-centrifugal');
const valStretch = document.getElementById('val-stretch');
const valCharge = document.getElementById('val-charge');

// Initialize Constants on UI
if(valH) valH.textContent = PLANCK_H.toExponential(3) + ' J⋅s';
if(valC) valC.textContent = SPEED_OF_LIGHT.toExponential(3) + ' m/s';

// Handle window resize
window.addEventListener('resize', () => {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
});

// -----------------------------------------------------------------------------
// Physics Sub-stepping & Time-Lapse Logic
// -----------------------------------------------------------------------------
const speedSlider = document.getElementById('speed-slider');
const speedVal = document.getElementById('speed-val');
let subSteps = 1;

if(speedSlider) {
    speedSlider.addEventListener('input', (e) => {
        subSteps = parseInt(e.target.value);
        speedVal.textContent = subSteps + "x";

        // As speed increases, decrease persistence very slightly to prevent total blowout,
        // but keep it high enough to form the solid n=1 orbital shell.
        fboMaterial.uniforms.opacity.value = 0.98 - (subSteps / 200.0) * 0.05;
    });
}

function updatePhysics(time, dtMultiplier) {
    // 2. Gravitational Attraction (Macroscopic Geometric Shadowing of Casimir Pressure)
    const distanceVector = new THREE.Vector3().subVectors(protonMesh.position, electronMesh.position);
    const distanceSq = distanceVector.lengthSq();
    const distance = Math.sqrt(distanceSq);

    const casimirShadowForce = (5.0 * 6.0) / (distanceSq + 0.1);
    const attractionForce = distanceVector.normalize().multiplyScalar(casimirShadowForce * 0.05);

    // 3. Mechanical Origin of Forces (Hydrogen Orbit)
    const coulombForceMagnitude = COULOMB_FORCE_CONSTANT / (distanceSq + 0.1);
    const coulombForce = distanceVector.clone().normalize().multiplyScalar(coulombForceMagnitude);

    const gravityMagnitude = GRAVITY_CONSTANT / (distanceSq + 0.1);
    const gravityForce = distanceVector.clone().normalize().multiplyScalar(gravityMagnitude);

    // Pilot-Wave Resonant Steering (Deterministic Precession)
    const radialDir = distanceVector.clone().normalize();
    const tangentialDir = electronVelocity.clone().normalize();
    const transverseDir = new THREE.Vector3().crossVectors(radialDir, tangentialDir).normalize();

    const wobbleStrength = Math.sin(time * 0.5) * 0.1;
    const pilotWaveResonance = transverseDir.multiplyScalar(wobbleStrength);

    // Total Acceleration Force
    const totalForce = coulombForce.clone().add(gravityForce).add(pilotWaveResonance);

    // Apply acceleration to velocities (a = F/m proxy), scaled by dt multiplier for stable sub-stepping
    const electronAcceleration = totalForce.clone().multiplyScalar((1.0 / ELECTRON_MASS_PROXY) * dtMultiplier);
    const protonAcceleration = totalForce.clone().negate().multiplyScalar((1.0 / PROTON_MASS_PROXY) * dtMultiplier);

    electronVelocity.add(electronAcceleration);
    protonVelocity.add(protonAcceleration);

    // Apply velocities to positions
    electronMesh.position.add(electronVelocity.clone().multiplyScalar(dtMultiplier));
    protonMesh.position.add(protonVelocity.clone().multiplyScalar(dtMultiplier));

    // Orient particles strictly perpendicular to their direction of motion
    const eLookTarget = electronMesh.position.clone().add(electronVelocity);
    electronMesh.lookAt(eLookTarget);
    const pLookTarget = protonMesh.position.clone().add(protonVelocity);
    protonMesh.lookAt(pLookTarget);

    return { distance, coulombForceMagnitude, casimirShadowForce, attractionForce, coulombForce };
}

// Render loop setup
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    const time = clock.getElapsedTime();

    // Update shader time uniforms
    vacuumFluidUniforms.uTime.value = time;
    electronMaterial.uniforms.time.value = time;
    protonMaterial.uniforms.time.value = time;

    // -------------------------------------------------------------------------
    // High-Fidelity Physics Sub-stepping & Continuous FBO Accumulation
    // -------------------------------------------------------------------------
    // To prevent "stuttering ghosts" at high speeds, we must render the scene
    // into the accumulation buffer at EVERY micro-step, creating a perfectly continuous optical blur.

    let physicsData = null;
    const configuredOpacity = fboMaterial.uniforms.opacity.value;

    // Disable auto-clear for continuous accumulation
    renderer.autoClear = false;

    for(let i = 0; i < subSteps; i++) {
        // 1. Update Physics Step (simulating 1 standard dt multiplier)
        physicsData = updatePhysics(time + (i * 0.016), 1.0);

        // 2. Update Dynamic Camera Tracking based on new positions
        const barycenter = new THREE.Vector3()
            .addScaledVector(electronMesh.position, ELECTRON_MASS_PROXY)
            .addScaledVector(protonMesh.position, PROTON_MASS_PROXY)
            .divideScalar(ELECTRON_MASS_PROXY + PROTON_MASS_PROXY);

        const clamp = (num, min, max) => Math.min(Math.max(num, min), max);
        const cameraDistance = clamp(physicsData.distance * 1.8, 60, 150);
        const targetCameraPos = new THREE.Vector3(barycenter.x, barycenter.y, barycenter.z + cameraDistance);

        // Instant/rapid camera tracking during sub-steps ensures it doesn't lag behind the fast orbit
        camera.position.lerp(targetCameraPos, 0.5);
        camera.lookAt(barycenter);

        // 3. Accumulate this specific micro-frame into the FBO
        renderer.setRenderTarget(renderTarget1);

        // Fade existing history on renderTarget1 using the tDiffuse of renderTarget2
        fboMaterial.uniforms.tDiffuse.value = renderTarget2.texture;
        renderer.render(fboScene, fboCamera);

        // Draw the current micro-step particle positions on top
        renderer.render(scene, camera);

        // Swap targets for the *next* micro-step in the loop
        let temp = renderTarget1;
        renderTarget1 = renderTarget2;
        renderTarget2 = temp;
    }

    // -------------------------------------------------------------------------
    // Final Output Pass to Screen (Once per requestAnimationFrame)
    // -------------------------------------------------------------------------
    renderer.setRenderTarget(null);
    renderer.clear(); // We do clear the actual screen buffer

    // We want to draw the *last accumulated* buffer (which is now sitting in renderTarget2
    // due to the final swap at the end of the sub-step loop) onto the screen.
    fboMaterial.uniforms.tDiffuse.value = renderTarget2.texture;

    // Force 1.0 opacity when drawing to screen so we don't accidentally fade the final image
    fboMaterial.uniforms.opacity.value = 1.0;
    renderer.render(fboScene, fboCamera);

    // Restore the fading opacity configured by the slider for the next total frame
    fboMaterial.uniforms.opacity.value = configuredOpacity;

    // Update UI Panel Real-Time Metrics using the final state
    if(physicsData) {
        if(valVel) valVel.textContent = electronVelocity.length().toFixed(4) + " c_proxy";
        const distancePm = physicsData.distance * (53.0 / BOHR_RADIUS_PROXY);
        if(valDist) valDist.textContent = distancePm.toFixed(2) + " pm";

        const casimirPressure = 178700 * (5 / physicsData.distance);
        const centrifugalMom = Math.pow(0.4, 2) / 5;

        if(valCasimir) valCasimir.textContent = casimirPressure.toFixed(0) + " N";
        if(valCentrifugal) valCentrifugal.textContent = centrifugalMom.toFixed(4) + " kg·m/s";
        if(valShadow) valShadow.textContent = physicsData.casimirShadowForce.toFixed(4) + " N";

        const acceleration = physicsData.attractionForce.length() + physicsData.coulombForce.length();
        if(valStretch) valStretch.textContent = (acceleration * 10).toFixed(4) + " λ";
        const rmsCharge = Math.sqrt(Math.pow(physicsData.coulombForceMagnitude, 2) / 2);
        if(valCharge) valCharge.textContent = rmsCharge.toFixed(4) + " E_rms";
    }
}

// Start simulation
animate();
