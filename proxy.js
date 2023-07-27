globalThis.console = {
  log: (msg) => {
    // TODO
    Deno.core.print(msg + "\n");
  },
};
const getCircularReplacer = () => {
  const seen = new WeakSet();
  return (key, value) => {
    if (typeof value === "object" && value !== null) {
      if (seen.has(value)) {
        return;
      }
      seen.add(value);
    }
    return value;
  };
};
console.log(JSON.stringify(globalThis, getCircularReplacer()));
(() => {
  class WaylandProxy {
    forward(context, message) {
      console.log("forwarding");
      context.send(message);
    }
  }
  return new WaylandProxy();
})();
