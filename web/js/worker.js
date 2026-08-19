/* Pyodide worker — runs edakit off the main thread. */

const PYODIDE_VERSION = "0.28.0";
const PYODIDE_CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
importScripts(`${PYODIDE_CDN}pyodide.js`);

let pyodide = null, bridge = null;
const post = (type, payload = {}) => self.postMessage({ type, ...payload });

/* A CDN mid-deploy can answer a .py request with an HTML 404 body. Writing
   that into the Python filesystem yields a SyntaxError pointing at GitHub's
   404 page, so verify the response before trusting it. */
async function fetchText(url, expectPython = false, attempts = 3) {
  let lastError;
  for (let i = 0; i < attempts; i++) {
    try {
      const res = await fetch(url, { cache: "no-cache" });
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
      const text = await res.text();
      if (/^\s*<(!doctype|html)/i.test(text)) {
        throw new Error(`${url} returned an HTML page, not the expected file`);
      }
      if (expectPython && !/^(""")|^(from )|^(import )|^(#)/m.test(text)) {
        throw new Error(`${url} does not look like Python source`);
      }
      return text;
    } catch (err) {
      lastError = err;
      if (i < attempts - 1) await new Promise((r) => setTimeout(r, 300 * 3 ** i));
    }
  }
  throw lastError;
}

async function boot() {
  post("boot", { stage: "Downloading Python runtime", pct: 0.05 });
  pyodide = await loadPyodide({
    indexURL: PYODIDE_CDN,
    stdout: (line) => post("log", { line }),
    stderr: (line) => post("log", { line, isError: true }),
  });

  post("boot", { stage: "Loading pandas and scipy", pct: 0.4 });
  await pyodide.loadPackage(["numpy", "pandas", "scipy"]);

  post("boot", { stage: "Installing edakit", pct: 0.85 });
  const manifest = JSON.parse(await fetchText("../py/manifest.json"));
  pyodide.FS.mkdirTree("/lib/edakit");
  await Promise.all(manifest.files.map(async (name) =>
    pyodide.FS.writeFile(`/lib/edakit/${name}`,
      await fetchText(`../py/edakit/${name}`, true))));
  pyodide.FS.writeFile("/lib/bridge.py", await fetchText("../py/bridge.py", true));

  pyodide.runPython(`import sys; sys.path.insert(0, "/lib")`);
  bridge = pyodide.pyimport("bridge");

  post("ready", {
    versions: pyodide.runPython(
      `import scipy, pandas, sys; f"Python {sys.version.split()[0]} · scipy {scipy.__version__} · pandas {pandas.__version__}"`),
  });
}

const HANDLERS = {
  analyse: ({ text, name }) => bridge.analyse(text, name),
};

self.onmessage = async ({ data }) => {
  const { id, action, ...args } = data;
  try {
    if (!bridge) throw new Error("Python runtime is still starting up.");
    post("result", { id, result: JSON.parse(await HANDLERS[action](args)) });
  } catch (err) {
    post("result", { id, result: { ok: false, error: String(err.message || err) } });
  }
};

boot().catch((err) => post("bootError", { error: String(err.message || err) }));
