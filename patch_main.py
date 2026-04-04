with open('main.js', 'r') as f:
    main_content = f.read()

math_functions = """
// -------------------------------------------------------------
// Global Simulation Parameters
// -------------------------------------------------------------
window.current_n = 4;
window.current_l = 3;
window.current_m = 1;
window.zpf_heat = 4.5;
window.vq_coupling = 20.0;

// -------------------------------------------------------------
// Hamiltonian Quantum Potential / Langevin Thermostat Functions
// -------------------------------------------------------------
function factorial(n) {
    let res = 1;
    for(let i=2; i<=n; i++) res *= i;
    return res;
}

function legendreP(l, m, x) {
    const m_abs = Math.abs(m);
    if (l === 0) return 1.0;
    if (l === 1) {
        if (m_abs === 0) return x;
        if (m_abs === 1) return -Math.sqrt(1.0 - x*x);
    }
    if (l === 2) {
        if (m_abs === 0) return 0.5 * (3.0 * x*x - 1.0);
        if (m_abs === 1) return -3.0 * x * Math.sqrt(1.0 - x*x);
        if (m_abs === 2) return 3.0 * (1.0 - x*x);
    }
    if (l === 3) {
        if (m_abs === 0) return 0.5 * (5.0 * Math.pow(x,3) - 3.0 * x);
        if (m_abs === 1) return -1.5 * (5.0 * x*x - 1.0) * Math.sqrt(1.0 - x*x);
        if (m_abs === 2) return 15.0 * x * (1.0 - x*x);
        if (m_abs === 3) return -15.0 * Math.pow(1.0 - x*x, 1.5);
    }
    if (l === 4) {
        if (m_abs === 0) return 0.125 * (35.0 * Math.pow(x,4) - 30.0 * x*x + 3.0);
        if (m_abs === 1) return -2.5 * (7.0 * Math.pow(x,3) - 3.0 * x) * Math.sqrt(1.0 - x*x);
        if (m_abs === 2) return 7.5 * (7.0 * x*x - 1.0) * (1.0 - x*x);
        if (m_abs === 3) return -105.0 * x * Math.pow(1.0 - x*x, 1.5);
        if (m_abs === 4) return 105.0 * Math.pow(1.0 - x*x, 2.0);
    }
    return 0.0;
}

function realSphericalHarmonic(l, m, theta, phi) {
    const m_abs = Math.abs(m);
    const N = Math.sqrt((2.0 * l + 1.0) / (4.0 * Math.PI) * factorial(l - m_abs) / factorial(l + m_abs));
    const P = legendreP(l, m_abs, Math.cos(theta));
    const CondonShortley = (m_abs % 2 === 1) ? -1 : 1;
    let Y = N * P * CondonShortley;

    if (m > 0) return Math.sqrt(2.0) * Y * Math.cos(m_abs * phi);
    if (m < 0) return Math.sqrt(2.0) * Y * Math.sin(m_abs * phi);
    return Y;
}

function calc_R_nl(r, n, l_val) {
    const a0 = 1.0;
    const rho = 2.0 * r / (n * a0);
    const norm = Math.sqrt(Math.pow(2.0 / (n * a0), 3) * factorial(n - l_val - 1) / (2.0 * n * factorial(n + l_val)));

    let laguerre = 0.0;
    if (n===1 && l_val===0) laguerre = 1.0;
    else if (n===2 && l_val===0) laguerre = 1.0 - 0.5 * rho;
    else if (n===2 && l_val===1) laguerre = 1.0;
    else if (n===3 && l_val===0) laguerre = 1.0 - rho + (1.0/6.0)*rho*rho;
    else if (n===3 && l_val===1) laguerre = 1.0 - (1.0/4.0)*rho;
    else if (n===3 && l_val===2) laguerre = 1.0;
    else if (n===4 && l_val===0) laguerre = 1.0 - (3.0/2.0)*rho + (1.0/2.0)*Math.pow(rho,2) - (1.0/24.0)*Math.pow(rho,3);
    else if (n===4 && l_val===1) laguerre = 1.0 - (3.0/5.0)*rho + (1.0/20.0)*Math.pow(rho,2);
    else if (n===4 && l_val===2) laguerre = 1.0 - (1.0/12.0)*rho;
    else if (n===4 && l_val===3) laguerre = 1.0;
    else if (n===5 && l_val===0) laguerre = 1.0 - (4.0/5.0)*rho + (4.0/25.0)*Math.pow(rho,2) - (4.0/375.0)*Math.pow(rho,3) + (1.0/3750.0)*Math.pow(rho,4);
    else if (n===5 && l_val===2) laguerre = 1.0 - (2.0/15.0)*rho + (1.0/210.0)*Math.pow(rho,2);
    else if (n===5 && l_val===3) laguerre = 1.0 - (1.0/20.0)*rho;
    else if (n===5 && l_val===4) laguerre = 1.0;

    return norm * Math.exp(-rho / 2.0) * Math.pow(rho, l_val) * laguerre;
}

function prob_density_cart(x, y, z, n, l_val, m_val) {
    let r = Math.sqrt(x*x + y*y + z*z);
    if (r < 1e-5) r = 1e-5;
    let theta = Math.acos(Math.max(-1.0, Math.min(1.0, z / r)));
    let phi = Math.atan2(y, x);
    let R = calc_R_nl(r, n, l_val);
    let Y = realSphericalHarmonic(l_val, m_val, theta, phi);
    return Math.pow(R * Y, 2);
}

function randomNormal() {
    let u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

"""

if "function prob_density_cart" not in main_content:
    main_content = main_content.replace("document.body.appendChild(renderer.domElement);", "document.body.appendChild(renderer.domElement);\n" + math_functions)
    with open('main.js', 'w') as f:
        f.write(main_content)
