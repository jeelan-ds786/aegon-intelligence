import { describe, expect, it } from "vitest";

import { packageName } from "../src/index.js";

describe("@aegon/indexer", () => {
  it("loads the package", () => {
    expect(packageName).toBe("@aegon/indexer");
  });
});