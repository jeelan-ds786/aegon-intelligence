import { describe, expect, it } from "vitest";

import { packageName } from "../src/index.js";

describe("@aegon/adopter-example", () => {
  it("loads the package", () => {
    expect(packageName).toBe("@aegon/adopter-example");
  });
});