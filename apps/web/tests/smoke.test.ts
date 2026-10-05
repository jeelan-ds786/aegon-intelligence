import { describe, expect, it } from "vitest";

import { packageName } from "../src/index.js";

describe("@aegon/web", () => {
  it("loads the package", () => {
    expect(packageName).toBe("@aegon/web");
  });
});