package com.bladevision.client;

import com.google.gson.JsonObject;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.phys.Vec3;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * BladeVision ground-truth exporter (Phase 0, lab instrumentation).
 *
 * <p>Read-only client-side mod. Each client tick it appends one JSON line describing the true
 * game state — self position/rotation/velocity/health and the nearest other player (the duel
 * opponent) — to a JSONL file, timestamped with the wall clock so it aligns with the Python
 * recorder's frame index (both processes share the machine clock; see docs/DESIGN.md §2.2).
 *
 * <p>This is what makes supervised perception cheap: it auto-labels the pixels with the real
 * entity geometry, and gives detectors a ground-truth reference. It NEVER emits input or
 * influences the game, and stays completely inert unless the {@code BLADEVISION_LAB} environment
 * variable is set — so it cannot accidentally run when shipped alongside anything else.
 */
public class BladeVisionClient implements ClientModInitializer {

	private static final String ENABLE_ENV = "BLADEVISION_LAB";
	private static final String DIR_ENV = "BLADEVISION_TRUTH_DIR";

	private BufferedWriter writer;

	@Override
	public void onInitializeClient() {
		if (System.getenv(ENABLE_ENV) == null) {
			// Lab gate: do nothing at all unless explicitly enabled.
			return;
		}
		openWriter();
		ClientTickEvents.END_CLIENT_TICK.register(this::onClientTick);
		ClientLifecycleEvents.CLIENT_STOPPING.register(c -> closeWriter());
	}

	private void openWriter() {
		try {
			String dir = System.getenv(DIR_ENV);
			Path base = (dir != null) ? Path.of(dir) : Path.of("bladevision");
			Files.createDirectories(base);
			String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
			Path out = base.resolve("truth_" + ts + ".jsonl");
			writer = Files.newBufferedWriter(out);
			System.out.println("[BladeVision] ground-truth logging to " + out.toAbsolutePath());
		} catch (IOException e) {
			System.err.println("[BladeVision] failed to open truth log: " + e);
		}
	}

	private void closeWriter() {
		if (writer == null) {
			return;
		}
		try {
			writer.flush();
			writer.close();
		} catch (IOException ignored) {
			// nothing actionable on shutdown
		} finally {
			writer = null;
		}
	}

	private void onClientTick(Minecraft client) {
		if (writer == null) {
			return;
		}
		LocalPlayer self = client.player;
		ClientLevel level = client.level;
		if (self == null || level == null) {
			return;
		}

		// The opponent in a 1v1 is the nearest other player.
		Player opponent = null;
		double bestSq = Double.MAX_VALUE;
		for (Player p : level.players()) {
			if (p == self) {
				continue;
			}
			double dSq = p.distanceToSqr(self);
			if (dSq < bestSq) {
				bestSq = dSq;
				opponent = p;
			}
		}

		try {
			writer.write(buildRecord(level, self, opponent));
			writer.write("\n");
			writer.flush();
		} catch (IOException ignored) {
			// a single dropped tick is not worth crashing the client
		}
	}

	private String buildRecord(ClientLevel level, LocalPlayer self, Player opponent) {
		JsonObject root = new JsonObject();
		root.addProperty("t_wall", System.currentTimeMillis());
		root.addProperty("t_nano", System.nanoTime());
		root.addProperty("tick", level.getGameTime());
		root.add("self", playerState(self, true));
		if (opponent != null) {
			JsonObject opp = playerState(opponent, false);
			opp.addProperty("dist", Math.sqrt(self.distanceToSqr(opponent)));
			root.add("opponent", opp);
		}
		return root.toString();
	}

	private JsonObject playerState(Player p, boolean isSelf) {
		JsonObject o = new JsonObject();
		Vec3 pos = p.position();
		Vec3 vel = p.getDeltaMovement();
		o.addProperty("x", pos.x);
		o.addProperty("y", pos.y);
		o.addProperty("z", pos.z);
		o.addProperty("vx", vel.x);
		o.addProperty("vy", vel.y);
		o.addProperty("vz", vel.z);
		o.addProperty("yaw", p.getYRot());
		o.addProperty("pitch", p.getXRot());
		o.addProperty("health", p.getHealth());
		o.addProperty("onGround", p.onGround());
		o.addProperty("sprinting", p.isSprinting());
		o.addProperty("hurtTime", p.hurtTime);      // >0 for the ticks after taking a hit
		o.addProperty("item", p.getMainHandItem().getDescriptionId());
		if (isSelf) {
			o.addProperty("food", p.getFoodData().getFoodLevel());
		}
		return o;
	}
}
