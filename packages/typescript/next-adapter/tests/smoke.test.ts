import { describe, expect, it } from "vitest";

import { packageName } from "../src/index.js";

describe("@aegon/next-adapter", () => {
  it("loads the package", () => {
    expect(packageName).toBe("@aegon/next-adapter");
  });
});