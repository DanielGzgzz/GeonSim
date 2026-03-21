// Topological Geon Physics Engine

// Initialize Three.js scene, camera, and renderer
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 50;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

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

// Render vacuum fluid backdrop
const vacuumGeometry = new THREE.PlaneGeometry(200, 200, 64, 64);
const vacuumPlane = new THREE.Mesh(vacuumGeometry, vacuumMaterial);
vacuumPlane.position.z = -50;
scene.add(vacuumPlane);

// -----------------------------------------------------------------------------
// Independent Wave Simulators (Topological Geons)
// -----------------------------------------------------------------------------
// Particles are NOT rigid bodies. They are emergent structures comprised of independent
// light waves circulating and constructively interfering along topological paths.
// We model each "geon" dynamically using an array of independent waves (spheres)
// governed by strict parametric topologies.

const PROTON_MASS_PROXY = 50.0;
const ELECTRON_MASS_PROXY = 1.0;

const BOHR_RADIUS_PROXY = 40.0;
const COULOMB_FORCE_CONSTANT = 400.0;
const GRAVITY_CONSTANT = 0.5;

// Common Barycenter tracking
const protonRadius = BOHR_RADIUS_PROXY * (ELECTRON_MASS_PROXY / (PROTON_MASS_PROXY + ELECTRON_MASS_PROXY));
const electronRadius = BOHR_RADIUS_PROXY * (PROTON_MASS_PROXY / (PROTON_MASS_PROXY + ELECTRON_MASS_PROXY));

// Core "Container" Meshes to track macroscopic position and velocity (these are invisible)
// The independent waves will orbit relative to these moving center points.
const electronMesh = new THREE.Mesh();
electronMesh.position.set(electronRadius, 0, 0);
scene.add(electronMesh);

const protonMesh = new THREE.Mesh();
protonMesh.position.set(-protonRadius, 0, 0);
scene.add(protonMesh);

// Helper function to generate an independent wave packet
function createWavePacket(color, size) {
    const geo = new THREE.SphereGeometry(size, 8, 8);
    const mat = new THREE.MeshBasicMaterial({ color: color, transparent: true, opacity: 0.8 });
    return new THREE.Mesh(geo, mat);
}

// -----------------------------------------------------------------------------
// Construct the Electron (4pi Möbius double-loop) via Independent Waves
// -----------------------------------------------------------------------------
const NUM_ELECTRON_WAVES = 150;
const ELECTRON_PATH_RADIUS = 5.0; // Large, loose
const ELECTRON_TUBE_RADIUS = 1.0;
const electronWaves = [];

for (let i = 0; i < NUM_ELECTRON_WAVES; i++) {
    const wave = createWavePacket(0x00ffff, 0.3);
    // Initialize phase (0 to 4pi for the full double loop)
    wave.userData.phaseU = (i / NUM_ELECTRON_WAVES) * 4 * Math.PI;
    // Cross-sectional phase (v)
    wave.userData.phaseV = Math.random() * 2 * Math.PI;

    electronWaves.push(wave);
    electronMesh.add(wave); // Add to container so it inherits macroscopic position/orientation
}

// -----------------------------------------------------------------------------
// Construct the Proton ((3,2)-trefoil knot) via Independent Waves
// -----------------------------------------------------------------------------
const NUM_PROTON_WAVES = 200;
const PROTON_PATH_RADIUS = 1.5; // Highly dense, small
const PROTON_TUBE_RADIUS = 0.6;
const protonWaves = [];

for (let i = 0; i < NUM_PROTON_WAVES; i++) {
    const wave = createWavePacket(0xff00ff, 0.15);
    // Initialize phase (0 to 2pi along the main trefoil curve)
    wave.userData.phaseU = (i / NUM_PROTON_WAVES) * 2 * Math.PI;
    // Cross-sectional phase (v)
    wave.userData.phaseV = Math.random() * 2 * Math.PI;

    protonWaves.push(wave);
    protonMesh.add(wave);
}

// Required orbital velocity for a circular orbit around the barycenter
// v = sqrt(F_inward * r_from_barycenter / m)
const forceAtDistance = COULOMB_FORCE_CONSTANT / Math.pow(BOHR_RADIUS_PROXY, 2);
const electronSpeed = Math.sqrt(forceAtDistance * electronRadius / ELECTRON_MASS_PROXY);
const protonSpeed = Math.sqrt(forceAtDistance * protonRadius / PROTON_MASS_PROXY);

const electronVelocity = new THREE.Vector3(0, electronSpeed, 0);
const protonVelocity = new THREE.Vector3(0, -protonSpeed, 0);

// Electron Trail Cloud (Macroscopic Vector Potential Field / Probability Cloud Proxy)
// The 2013 quantum microscope image of Hydrogen shows a "probability cloud".
// We simulate this deterministically by rendering the time-averaged path history
// of the electron using a large particle system.
const cloudLength = 3000;
const cloudPositions = new Float32Array(cloudLength * 3);
// Initialize all points off-screen or at origin
for(let i=0; i<cloudLength*3; i++) cloudPositions[i] = 10000;

const cloudGeometry = new THREE.BufferGeometry();
cloudGeometry.setAttribute('position', new THREE.BufferAttribute(cloudPositions, 3));

// We use Points instead of a continuous Line to prevent drawing massive artifacts
// across the screen and to better simulate a cumulative density "cloud"
const cloudMaterial = new THREE.PointsMaterial({
    color: 0x00ffff,
    size: 0.5,
    transparent: true,
    opacity: 0.4,
    blending: THREE.AdditiveBlending
});
const electronCloud = new THREE.Points(cloudGeometry, cloudMaterial);
scene.add(electronCloud);
let cloudIndex = 0;

// Proton Trail (Shows barycenter orbit explicitly)
// We avoid a continuous Line buffer artifact by shifting an array linearly rather than circularly.
const protonTrailLength = 200; // Shorter tail for cleaner look
const protonTrailPositions = new Float32Array(protonTrailLength * 3);
const protonTrailGeometry = new THREE.BufferGeometry();
protonTrailGeometry.setAttribute('position', new THREE.BufferAttribute(protonTrailPositions, 3));
const protonTrailMaterial = new THREE.LineBasicMaterial({
    color: 0xff00ff,
    transparent: true,
    opacity: 0.8,
    blending: THREE.AdditiveBlending
});
const protonTrail = new THREE.Line(protonTrailGeometry, protonTrailMaterial);
scene.add(protonTrail);
let protonTrailCurrentLength = 0;


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

// Render loop setup (to be expanded)
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    const time = clock.getElapsedTime();

    // Update shader time uniforms
    vacuumFluidUniforms.uTime.value = time;

    // -------------------------------------------------------------------------
    // Independent Wave Propagation Update (Internal Light Speed)
    // -------------------------------------------------------------------------
    const electronWaveSpeed = 15.0 * 0.016; // internal c proxy per frame
    const protonWaveSpeed = 3.0 * 0.016;

    // Update individual electron waves (4pi Möbius double-loop topology)
    electronWaves.forEach((wave) => {
        wave.userData.phaseU += electronWaveSpeed;
        if (wave.userData.phaseU > 4 * Math.PI) wave.userData.phaseU -= 4 * Math.PI;

        // Minor spiral/vortex component along v for fluid look
        wave.userData.phaseV += electronWaveSpeed * 2.0;

        const u = wave.userData.phaseU;
        const v = wave.userData.phaseV;
        const twist = u / 2;

        const cx = ELECTRON_TUBE_RADIUS * Math.cos(v);
        const cy = ELECTRON_TUBE_RADIUS * Math.sin(v);

        const twistedX = cx * Math.cos(twist) - cy * Math.sin(twist);
        const twistedY = cx * Math.sin(twist) + cy * Math.cos(twist);

        wave.position.x = (ELECTRON_PATH_RADIUS + twistedX) * Math.cos(u / 2);
        wave.position.y = (ELECTRON_PATH_RADIUS + twistedX) * Math.sin(u / 2);
        wave.position.z = twistedY;
    });

    // Update individual proton waves ((3,2)-trefoil knot topology)
    protonWaves.forEach((wave) => {
        wave.userData.phaseU += protonWaveSpeed;
        if (wave.userData.phaseU > 2 * Math.PI) wave.userData.phaseU -= 2 * Math.PI;

        wave.userData.phaseV += protonWaveSpeed * 3.0;

        const u = wave.userData.phaseU;
        const v = wave.userData.phaseV;

        // Base (3,2) trefoil math
        const px = Math.sin(u) + 2 * Math.sin(2 * u);
        const py = Math.cos(u) - 2 * Math.cos(2 * u);
        const pz = -Math.sin(3 * u);

        // Add tube thickness
        const rad = PROTON_PATH_RADIUS * 0.5;
        wave.position.x = px * rad + PROTON_TUBE_RADIUS * Math.cos(v) * Math.cos(u);
        wave.position.y = py * rad + PROTON_TUBE_RADIUS * Math.cos(v) * Math.sin(u);
        wave.position.z = pz * rad + PROTON_TUBE_RADIUS * Math.sin(v);
    });

    // -------------------------------------------------------------------------
    // Relativistic Kinematics & Electromagnetism Update
    // -------------------------------------------------------------------------

    // 2. Gravitational Attraction (Macroscopic Geometric Shadowing of Casimir Pressure)
    // Calculate distance between electron and proton knots
    const distanceVector = new THREE.Vector3().subVectors(protonMesh.position, electronMesh.position);
    const distanceSq = distanceVector.lengthSq();
    const distance = Math.sqrt(distanceSq);

    // Casimir shadow pressure proxy: inverse square of distance, modified by knot geometry cross-sections
    // Approximation: larger geometry = more shadowing = stronger pull
    const casimirShadowForce = (5.0 * 6.0) / (distanceSq + 0.1);
    const attractionForce = distanceVector.normalize().multiplyScalar(casimirShadowForce * 0.05);

    // 3. Mechanical Origin of Forces (Hydrogen Orbit)

    // Electromagnetism (Gauss Shell Illusion via RMS Radial Vectors)
    // The Coulomb force proxy (inward radial flux projection)
    // Using distanceSq instead of distance prevents the math from exploding, but we must use the configured constant.
    const coulombForceMagnitude = COULOMB_FORCE_CONSTANT / (distanceSq + 0.1);
    const coulombForce = distanceVector.clone().normalize().multiplyScalar(coulombForceMagnitude);

    // Gravity (Mutual Casimir Shadowing)
    const gravityMagnitude = GRAVITY_CONSTANT / (distanceSq + 0.1);
    const gravityForce = distanceVector.clone().normalize().multiplyScalar(gravityMagnitude);

    // -------------------------------------------------------------------------
    // Pilot-Wave Resonant Steering (Probabilistic but Deterministic Cloud)
    // -------------------------------------------------------------------------
    // The electron is steered deterministically by its extended vector potential.
    // To prevent the orbit from exploding into deep space, the wobble MUST strictly be
    // tangential/transverse to the radius vector, NOT radial.
    // We compute a stable transverse wobble using cross products.
    const radialDir = distanceVector.clone().normalize();
    const tangentialDir = electronVelocity.clone().normalize();
    const transverseDir = new THREE.Vector3().crossVectors(radialDir, tangentialDir).normalize();

    // Apply a slowly oscillating perturbation along the transverse axis
    const wobbleStrength = Math.sin(time * 0.5) * 0.1;
    const pilotWaveResonance = transverseDir.multiplyScalar(wobbleStrength);

    // Total Acceleration Force
    const totalForce = coulombForce.clone().add(gravityForce).add(pilotWaveResonance);

    // Apply acceleration to velocities (a = F/m proxy)
    // Proton is 1836x more massive, so it accelerates much less.
    const electronAcceleration = totalForce.clone().multiplyScalar(1.0 / ELECTRON_MASS_PROXY);
    const protonAcceleration = totalForce.clone().negate().multiplyScalar(1.0 / PROTON_MASS_PROXY);

    electronVelocity.add(electronAcceleration);
    protonVelocity.add(protonAcceleration);

    // Apply velocities to positions (Kinematics)
    electronMesh.position.add(electronVelocity);
    protonMesh.position.add(protonVelocity);

    // Orient particles strictly perpendicular to their direction of motion
    // This perfectly mimics fluid vortex ring dynamics (smoke rings) traversing the Casimir vacuum
    const eLookTarget = electronMesh.position.clone().add(electronVelocity);
    electronMesh.lookAt(eLookTarget);

    const pLookTarget = protonMesh.position.clone().add(protonVelocity);
    protonMesh.lookAt(pLookTarget);

    // Update Electron Probability Cloud Trail
    const positions = electronCloud.geometry.attributes.position.array;
    positions[cloudIndex * 3] = electronMesh.position.x;
    positions[cloudIndex * 3 + 1] = electronMesh.position.y;
    positions[cloudIndex * 3 + 2] = electronMesh.position.z;

    cloudIndex = (cloudIndex + 1) % cloudLength;
    electronCloud.geometry.attributes.position.needsUpdate = true;

    // Update Proton Orbit Trail (Linear shift instead of circular buffer to prevent artifacts)
    const pPositions = protonTrail.geometry.attributes.position.array;

    // Shift all points back one slot
    for (let i = protonTrailLength - 1; i > 0; i--) {
        pPositions[i * 3] = pPositions[(i - 1) * 3];
        pPositions[i * 3 + 1] = pPositions[(i - 1) * 3 + 1];
        pPositions[i * 3 + 2] = pPositions[(i - 1) * 3 + 2];
    }

    // Add current position to head
    pPositions[0] = protonMesh.position.x;
    pPositions[1] = protonMesh.position.y;
    pPositions[2] = protonMesh.position.z;

    if (protonTrailCurrentLength < protonTrailLength) {
        protonTrailCurrentLength++;
    }

    // Only draw the points we have accumulated so far
    protonTrail.geometry.setDrawRange(0, protonTrailCurrentLength);
    protonTrail.geometry.attributes.position.needsUpdate = true;

    // Dynamic camera tracking: Focus on the true Barycenter (Center of Mass) of the system
    // Calculate the weighted center of mass dynamically
    const barycenter = new THREE.Vector3()
        .addScaledVector(electronMesh.position, ELECTRON_MASS_PROXY)
        .addScaledVector(protonMesh.position, PROTON_MASS_PROXY)
        .divideScalar(ELECTRON_MASS_PROXY + PROTON_MASS_PROXY);

    // Pull back enough to see the orbiting Electron. We clamp it so it never flies away.
    const clamp = (num, min, max) => Math.min(Math.max(num, min), max);
    const cameraDistance = clamp(distance * 1.8, 60, 150);

    // Slow, stable camera tracking focused on the barycenter
    const targetCameraPos = new THREE.Vector3(barycenter.x, barycenter.y, barycenter.z + cameraDistance);
    camera.position.lerp(targetCameraPos, 0.02);
    camera.lookAt(barycenter);

    // Update Coulomb Arrow Visuals (Cyan)
    const coulombDir = distanceVector.clone().normalize();
    coulombArrowElectron.position.copy(electronMesh.position);
    coulombArrowElectron.setDirection(coulombDir);
    coulombArrowElectron.setLength(Math.max(1, coulombForceMagnitude * 30));

    coulombArrowProton.position.copy(protonMesh.position);
    coulombArrowProton.setDirection(coulombDir.clone().negate());
    coulombArrowProton.setLength(Math.max(1, coulombForceMagnitude * 30));

    // Update UI Panel Real-Time Metrics
    if(valVel) valVel.textContent = electronVelocity.length().toFixed(4) + " c_proxy";

    // Convert WebGL distance (40 proxy) to realistic pm (picometers) proxy. (Bohr radius ~53pm)
    const distancePm = distance * (53.0 / BOHR_RADIUS_PROXY);
    if(valDist) valDist.textContent = distancePm.toFixed(2) + " pm";

    // Casimir vacuum pressure vs outward centrifugal momentum balances the "mass"
    const radiusEProxy = 5;
    const casimirPressure = 178700 * (5 / distance); // N proxy
    const centrifugalMom = Math.pow(0.4, 2) / 5; // Fixed proxy for UI matching slowed down rotation concept

    if(valCasimir) valCasimir.textContent = casimirPressure.toFixed(0) + " N";
    if(valCentrifugal) valCentrifugal.textContent = centrifugalMom.toFixed(4) + " kg·m/s";

    // Shadowing and inertial resistance
    if(valShadow) valShadow.textContent = casimirShadowForce.toFixed(4) + " N";

    // Inertial helical stretch proxy: proportional to combined acceleration forces
    const acceleration = attractionForce.length() + coulombForce.length();
    if(valStretch) valStretch.textContent = (acceleration * 10).toFixed(4) + " λ";

    // Effective RMS charge integration proxy
    // Root mean square of internal varying radial vectors projecting the Gauss Shell
    const rmsCharge = Math.sqrt(Math.pow(coulombForceMagnitude, 2) / 2);
    if(valCharge) valCharge.textContent = rmsCharge.toFixed(4) + " E_rms";

    renderer.render(scene, camera);
}

// Start simulation
animate();
