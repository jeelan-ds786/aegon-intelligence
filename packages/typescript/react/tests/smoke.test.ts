import { describe, expect, it } from "vitest";

import { packageName } from "../src/index.js";

describe("@aegon/react", () => {
  it("loads the package", () => {
    expect(packageName).toBe("@aegon/react");
  });
});