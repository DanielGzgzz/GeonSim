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
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
directionalLight.position.set(0, 0, 50); // Pointing at the center from camera z pos
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

// Electron Topology: 4pi Möbius Double-Loop
function createMobiusDoubleLoopGeometry(radius, minorRadius, tubeThickness, tubularSegments, radialSegments) {
    const curve = new THREE.Curve();
    curve.getPoint = function (t, optionalTarget = new THREE.Vector3()) {
        const u = t * Math.PI * 4;
        const v = t * Math.PI * 2; // For a true double loop, the twist matches the orbit.
        const x = (radius + minorRadius * Math.cos(v)) * Math.cos(u);
        const y = (radius + minorRadius * Math.cos(v)) * Math.sin(u);
        const z = minorRadius * Math.sin(v);
        return optionalTarget.set(x, y, z);
    };

    const geometry = new THREE.TubeGeometry(curve, tubularSegments, tubeThickness, radialSegments, true); // Should be closed to be unbroken

    // Apply a strict 4pi (720 degree) geometric twist to the vertices along the path
    const positionAttribute = geometry.attributes.position;
    const vertex = new THREE.Vector3();
    const center = new THREE.Vector3();

    // The TubeGeometry distributes vertices evenly along the tubularSegments.
    // For each tubular segment, there are `radialSegments + 1` vertices.
    for (let i = 0; i <= tubularSegments; i++) {
        // Find the center point of this segment on the curve
        const t = i / tubularSegments;
        curve.getPoint(t, center);

        // The total twist angle for this segment (up to 4*PI total)
        const angle = t * Math.PI * 4;

        // The Frenet frame (tangent) at this point to rotate around
        const tangent = curve.getTangent(t).normalize();

        for (let j = 0; j <= radialSegments; j++) {
            const index = i * (radialSegments + 1) + j;
            if (index >= positionAttribute.count) continue;

            vertex.fromBufferAttribute(positionAttribute, index);

            // Translate vertex to origin relative to the center of the tube
            vertex.sub(center);

            // Apply the twist rotation around the path's tangent vector
            vertex.applyAxisAngle(tangent, angle);

            // Translate back
            vertex.add(center);

            positionAttribute.setXYZ(index, vertex.x, vertex.y, vertex.z);
        }
    }

    // Recompute normals since we drastically altered the surface
    geometry.computeVertexNormals();

    return geometry;
}

// Proton Topology: (3,2) Trefoil Knot
function createTrefoilKnotGeometry(radius, tubeThickness, tubularSegments, radialSegments) {
    const curve = new THREE.Curve();
    curve.getPoint = function (t, optionalTarget = new THREE.Vector3()) {
        const u = t * Math.PI * 2;
        // Standard closed trefoil knot
        const x = radius * (Math.sin(u) + 2 * Math.sin(2 * u)) * 0.3;
        const y = radius * (Math.cos(u) - 2 * Math.cos(2 * u)) * 0.3;
        const z = radius * -Math.sin(3 * u) * 0.3;
        return optionalTarget.set(x, y, z);
    };

    return new THREE.TubeGeometry(curve, tubularSegments, tubeThickness, radialSegments, true);
}

// -----------------------------------------------------------------------------
// Visual Representation (Materiality & Tension)
// -----------------------------------------------------------------------------
// The Liquid Metal Material: hyper-dense fluid reacting to pressure.
const liquidMetalMat = new THREE.MeshPhysicalMaterial({
    color: 0x888888,     // Base chrome
    emissive: 0x00E676,  // Mint green internal energy glow
    emissiveIntensity: 0.2, // Spikes during photon absorption
    metalness: 1.0,      // Pure reflection
    roughness: 0.05,     // Flawless surface
    clearcoat: 1.0,
    transparent: false,
    wireframe: false
});

const electronMaterial = liquidMetalMat.clone();
const protonMaterial = liquidMetalMat.clone();

// -----------------------------------------------------------------------------
// Hydrogen Atom Model (Topological Geon Framework)
// -----------------------------------------------------------------------------
const PROTON_MASS_PROXY = 50.0;
const ELECTRON_MASS_PROXY = 1.0;

const BOHR_RADIUS_PROXY = 40.0;
// We will dynamically calculate the Coulomb force proxy to exactly balance the
// required relativistic orbital speed (alpha * c).
// Gravity is kept extremely weak (negligible) as a generic macroscopic proxy.
const GRAVITY_CONSTANT = 0.05;

// Common Barycenter tracking
const protonRadius = BOHR_RADIUS_PROXY * (ELECTRON_MASS_PROXY / (PROTON_MASS_PROXY + ELECTRON_MASS_PROXY));
const electronRadius = BOHR_RADIUS_PROXY * (PROTON_MASS_PROXY / (PROTON_MASS_PROXY + ELECTRON_MASS_PROXY));

// -----------------------------------------------------------------------------
// Real-Time Topological Physics Engine (Dynamic Geometries)
// -----------------------------------------------------------------------------
// System Constants
const R_0 = 52.9;
const K_VAC = 52.9;
const DAMPING = 0.85; // Vacuum kinematic viscosity
const FLECHETTE_LIMIT = 200.0; // Topological breaking point

// Geon State Variables (Electron)
let electron_r = R_0;
let electron_velocity_r = 0.0;
let electron_E_current = 1.0;

// Geon State Variables (Proton)
let proton_r = R_0 * 0.1; // Proton is significantly smaller, scaled by 0.1
let proton_velocity_r = 0.0;
let proton_E_current = 1.0 * 0.1; // Energy scaled for equilibrium with smaller radius

// The Proton: A highly compressed, dense, tiny (3,2)-trefoil knot
const protonGeometry = createTrefoilKnotGeometry(1.0, 0.15, 256, 32);
protonGeometry.computeVertexNormals();
const protonMesh = new THREE.Mesh(protonGeometry, protonMaterial);
protonMesh.position.set(-protonRadius, 0, 0);
scene.add(protonMesh);

// The Electron: A loose, large, extended 4pi Möbius double-loop
// major radius: 1.0, minor loop radius: 0.3, tube thickness: 0.02 (high-tension wire), radial segments: 4 (sharp cornered cross-section)
const electronGeometry = createMobiusDoubleLoopGeometry(1.0, 0.3, 0.02, 512, 4);
electronGeometry.computeVertexNormals();
const electronMesh = new THREE.Mesh(electronGeometry, electronMaterial);
electronMesh.position.set(electronRadius, 0, 0);
scene.add(electronMesh);

function triggerFlechetteDestruction(mesh) {
    mesh.visible = false;
    // Further logic for linear light rays could be implemented here
    console.warn("FLECHETTE LIMIT REACHED: Topological structure broken.");
}

function updateGeonRadii() {
    // 5. Update Solid Geometry Scale
    const scale = electron_r / R_0;
    electronMesh.scale.set(scale, scale, 1.0);

    const protonScale = proton_r / R_0;
    protonMesh.scale.set(protonScale, protonScale, 1.0);
}
// Initialize the starting scales
updateGeonRadii();

// -----------------------------------------------------------------------------
// Relativistic Kinematics (Barycenter & Fine Structure Constant)
// -----------------------------------------------------------------------------
// In the Topological Geon framework, the internal trapped light moves at c.
// However, the macroscopic geometric track (the particle itself) orbits the
// nucleus at roughly v = alpha * c, where alpha (Fine Structure Constant) ≈ 1/137.
const ALPHA = 1.0 / 137.036;

// We set an arbitrary visual proxy for 'c' to keep the simulation visually readable.
// This is independent of the shader's internal 'wave' speed.
const ORBITAL_C_PROXY = 50.0;

const electronSpeed = ORBITAL_C_PROXY * ALPHA;
// Conservation of momentum for the barycenter orbit: m1 * v1 = m2 * v2
const protonSpeed = electronSpeed * (ELECTRON_MASS_PROXY / PROTON_MASS_PROXY);

// To achieve a perfectly stable circular orbit at these slow relativistic speeds,
// the centripetal force required is F_c = m * v^2 / r.
// Our total inward force is F_coulomb + F_gravity.
// We configure COULOMB_FORCE_CONSTANT so that: (k / R^2) = m * v^2 / r
// Therefore: k = (m * v^2 * R^2) / r. (Subtracting gravity contribution).
const requiredElectronForce = (ELECTRON_MASS_PROXY * Math.pow(electronSpeed, 2)) / electronRadius;
// requiredTotalForce * distance^2 = K_total.
const K_total = requiredElectronForce * Math.pow(BOHR_RADIUS_PROXY, 2);
const COULOMB_FORCE_CONSTANT = K_total - GRAVITY_CONSTANT;

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

// -----------------------------------------------------------------------------
// Interactive Trigger: Photon Absorption
// -----------------------------------------------------------------------------
const firePhotonBtn = document.getElementById('fire-photon-btn');
const PHOTON_ENERGY_PROXY = 1.0; // Amount of energy the electron absorbs, doubling to 2.0

// Free Photon Visual Entity
const photonGeometry = new THREE.SphereGeometry(1.5, 16, 16);
const photonMaterial = new THREE.MeshBasicMaterial({ color: 0xffff00 });
const photonMesh = new THREE.Mesh(photonGeometry, photonMaterial);
photonMesh.visible = false;
scene.add(photonMesh);

let isPhotonActive = false;
const PHOTON_SPEED = ORBITAL_C_PROXY * 1.0; // 1.0 c
let photonVelocity = new THREE.Vector3();

if(firePhotonBtn) {
    firePhotonBtn.addEventListener('click', () => {
        if(isPhotonActive) return;

        // Launch Photon from outside the orbit towards the electron
        photonMesh.position.copy(electronMesh.position).add(new THREE.Vector3(150, 0, 0)); // Start 150 units away
        photonVelocity = new THREE.Vector3(-1, 0, 0).normalize().multiplyScalar(PHOTON_SPEED);
        photonMesh.visible = true;
        isPhotonActive = true;

        firePhotonBtn.textContent = "Photon Approaching...";
        firePhotonBtn.style.color = "yellow";
        firePhotonBtn.disabled = true;
    });
}

function updatePhysics(time, dtMultiplier) {
    // 1. Photon Travel Logic
    if (isPhotonActive) {
        // Since physics is sub-stepped, we move the photon in smaller increments
        const photonStep = photonVelocity.clone().multiplyScalar(0.016 * dtMultiplier);
        photonMesh.position.add(photonStep);

        // Collision Detection with Electron
        if (photonMesh.position.distanceTo(electronMesh.position) < 5.0) {
            isPhotonActive = false;
            photonMesh.visible = false;

            // 1. The Swell: Add energy. The outward centrifugal momentum (E/r) instantly spikes.
            electron_E_current += PHOTON_ENERGY_PROXY; // Now hits 2.0
            electronMaterial.emissiveIntensity = 2.0;

            firePhotonBtn.textContent = "Electron Excited!";
            firePhotonBtn.style.color = "#ffaa00";

            // 2. The Emission: Refractive index delay causes the electron to eventually eject the photon.
            setTimeout(() => {
                electron_E_current -= PHOTON_ENERGY_PROXY;
                firePhotonBtn.textContent = "Fire Photon (Absorb Energy)";
                firePhotonBtn.style.color = "#0ff";
                firePhotonBtn.disabled = false;

                // Visual emission: send photon away
                photonMesh.position.copy(electronMesh.position);
                photonVelocity = electronVelocity.clone().normalize().multiplyScalar(PHOTON_SPEED); // Emit tangentially
                photonMesh.visible = true;

                setTimeout(() => { photonMesh.visible = false; }, 1000); // Hide after a second

            }, 1500); // 1.5 seconds in excited state before emitting and crushing back down
        }
    }

    // Smoothly tween emissive intensity back to base level
    if (electronMaterial.emissiveIntensity > 0.2) {
        electronMaterial.emissiveIntensity -= 0.05 * dtMultiplier;
        if (electronMaterial.emissiveIntensity < 0.2) electronMaterial.emissiveIntensity = 0.2;
    }
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

    // Hard-Clamp the Velocity to roughly v = alpha * c
    const maxVelocity = ORBITAL_C_PROXY * ALPHA;
    if (electronVelocity.length() > maxVelocity) {
        electronVelocity.setLength(maxVelocity);
    }
    const maxProtonVelocity = maxVelocity * (ELECTRON_MASS_PROXY / PROTON_MASS_PROXY);
    if (protonVelocity.length() > maxProtonVelocity) {
        protonVelocity.setLength(maxProtonVelocity);
    }

    // Apply velocities to positions
    electronMesh.position.add(electronVelocity.clone().multiplyScalar(dtMultiplier));
    protonMesh.position.add(protonVelocity.clone().multiplyScalar(dtMultiplier));

    // Orient particles strictly perpendicular to their direction of motion
    const eLookTarget = electronMesh.position.clone().add(electronVelocity);
    electronMesh.lookAt(eLookTarget);
    const pLookTarget = protonMesh.position.clone().add(protonVelocity);
    protonMesh.lookAt(pLookTarget);

    // -------------------------------------------------------------------------
    // 4. Dynamic Equilibrium: Internal Energy vs Vacuum Crush
    // -------------------------------------------------------------------------
    // 1. Clamp dt to prevent integration explosions if framerate drops
    const dt = Math.min(0.016 * dtMultiplier, 0.032);

    // Electron Radius Update
    // 2. Calculate Opposing Fluid Forces
    let e_F_out = electron_E_current / electron_r;
    let e_F_in = K_VAC / (electron_r * electron_r);
    let e_F_net = e_F_out - e_F_in;

    // 3. Euler Integration with strict vacuum damping
    electron_velocity_r += e_F_net * dt;
    electron_velocity_r *= DAMPING;
    electron_r += electron_velocity_r * dt;

    // 4. The Flechette Fail-State
    if (electron_r > FLECHETTE_LIMIT && electronMesh.visible) {
        triggerFlechetteDestruction(electronMesh);
    }

    // Proton Radius Update (Using K_VAC scaled appropriately for proton)
    // We used proton_E_current = 0.1, proton_r = R_0 * 0.1
    // For F_out = F_in => E/R = K_VAC_P / R^2 => 0.1 / (R_0 * 0.1) = K_VAC_P / (R_0 * 0.1)^2
    // 1 / R_0 = K_VAC_P / (R_0^2 * 0.01) => K_VAC_P = R_0 * 0.01
    const K_VAC_P = K_VAC * 0.01;
    let p_F_out = proton_E_current / proton_r;
    let p_F_in = K_VAC_P / (proton_r * proton_r);
    let p_F_net = p_F_out - p_F_in;

    proton_velocity_r += p_F_net * dt;
    proton_velocity_r *= DAMPING;
    proton_r += proton_velocity_r * dt;

    if (proton_r > FLECHETTE_LIMIT && protonMesh.visible) {
        triggerFlechetteDestruction(protonMesh);
    }

    updateGeonRadii();

    return { distance, coulombForceMagnitude, casimirShadowForce, attractionForce, coulombForce };
}

// Render loop setup
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    const time = clock.getElapsedTime();

    // Update shader time uniforms
    vacuumFluidUniforms.uTime.value = time;

    // -------------------------------------------------------------------------
    // Vanilla Geometry Test Rendering
    // -------------------------------------------------------------------------

    let physicsData = null;
    renderer.autoClear = true; // Auto clear for basic test

    for(let i = 0; i < subSteps; i++) {
        // 1. Update Physics Step
        physicsData = updatePhysics(time + (i * 0.016), 1.0);
    }

    // -------------------------------------------------------------------------
    // Final Output Pass to Screen
    // -------------------------------------------------------------------------
    if(physicsData) {
        const barycenter = new THREE.Vector3()
            .addScaledVector(electronMesh.position, ELECTRON_MASS_PROXY)
            .addScaledVector(protonMesh.position, PROTON_MASS_PROXY)
            .divideScalar(ELECTRON_MASS_PROXY + PROTON_MASS_PROXY);

        const clamp = (num, min, max) => Math.min(Math.max(num, min), max);
        const cameraDistance = clamp(physicsData.distance * 1.8, 60, 150);
        const targetCameraPos = new THREE.Vector3(barycenter.x, barycenter.y, barycenter.z + cameraDistance);

        camera.position.lerp(targetCameraPos, 0.05);
        camera.lookAt(barycenter);
    }

    // Render directly to screen
    renderer.setRenderTarget(null);
    renderer.render(scene, camera);

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
