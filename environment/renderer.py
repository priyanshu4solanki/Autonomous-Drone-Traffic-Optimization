import pygame


class Renderer:
    def __init__(self, env, cell_size=28, sidebar_width=440, fps=3, policy_type="CNN"):
        self.env, self.cell, self.sidebar_w, self.fps, self.running = env, cell_size, sidebar_width, fps, True
        self.policy_type = policy_type
        pygame.init()

        self.w = env.city.width * cell_size
        self.grid_h = env.city.height * cell_size
        self.h = max(self.grid_h, 700 + 25 * env.num_drones)
        self.screen = pygame.display.set_mode((self.w + sidebar_width, self.h))
        pygame.display.set_caption("Autonomous Drone Traffic Optimization")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Segoe UI", 20, bold=True)
        self.small = pygame.font.SysFont("Segoe UI", 16)
        self.tiny = pygame.font.SysFont("Segoe UI", 14)
        self.selected_drone = 0
        self.actions = None
        self.action_probabilities = None
        self.paused = False
        self.single_step = False
        self.result_summary = None

        self.trails = {i: [] for i in range(env.num_drones)}
        self.colors = [
            (99, 102, 241),   # Indigo
            (239, 68, 68),    # Red
            (168, 85, 247),   # Purple
            (245, 158, 11),   # Amber
            (14, 165, 233),   # Sky Blue
            (34, 197, 94),    # Green
            (236, 72, 153),   # Pink
            (20, 184, 166),   # Teal
            (250, 204, 21),   # Yellow
            (129, 140, 248),  # Light Indigo
        ]

    def render(self, actions=None, action_probabilities=None):
        if not self.running:
            return False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.KEYDOWN and pygame.K_1 <= event.key <= pygame.K_9:
                selected = event.key - pygame.K_1
                if selected < self.env.num_drones:
                    self.selected_drone = selected
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.single_step = True
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                self.fps = min(30, self.fps + 1)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_MINUS:
                self.fps = max(1, self.fps - 1)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                self.selected_drone = (self.selected_drone + 1) % self.env.num_drones

        self.actions = actions
        self.action_probabilities = action_probabilities

        # Backgrounds
        self.screen.fill((18, 20, 26))
        pygame.draw.rect(self.screen, (28, 32, 44), (self.w, 0, self.sidebar_w, self.h))

        self._grid()
        self._city()
        self._drones()
        self._sidebar()

        pygame.display.flip()
        self.clock.tick(self.fps)
        if self.result_summary is not None:
            return False
        if self.paused and not self.single_step:
            return False
        if self.single_step:
            self.single_step = False
            self.paused = True
        return self.running

    def _grid(self):
        for x in range(self.env.city.width + 1):
            pygame.draw.line(self.screen, (28, 32, 44), (x * self.cell, 0), (x * self.cell, self.grid_h))
        for y in range(self.env.city.height + 1):
            pygame.draw.line(self.screen, (28, 32, 44), (0, y * self.cell), (self.w, y * self.cell))

    def _city(self):
        city = self.env.city
        for r in city.buildings:
            pygame.draw.rect(
                self.screen, (76, 84, 102),
                (r.x1 * self.cell + 2, r.y1 * self.cell + 2, (r.x2 - r.x1 + 1) * self.cell - 4, (r.y2 - r.y1 + 1) * self.cell - 4),
                border_radius=4
            )
        for r in city.no_fly_zones:
            pygame.draw.rect(
                self.screen, (120, 48, 58),
                (r.x1 * self.cell + 2, r.y1 * self.cell + 2, (r.x2 - r.x1 + 1) * self.cell - 4, (r.y2 - r.y1 + 1) * self.cell - 4),
                border_radius=4
            )

        for p, col, lbl in [(p, (59, 130, 246), "W") for p in city.warehouses] + \
                            [(p, (34, 197, 94), "D") for p in city.delivery_points] + \
                            [(p, (234, 179, 8), "C") for p in city.charging_stations]:
            self._point(p, col, lbl)

    def _point(self, p, color, label):
        x, y = p.x * self.cell + self.cell // 2, p.y * self.cell + self.cell // 2
        pygame.draw.circle(self.screen, color, (x, y), max(8, self.cell // 3))
        text = self.tiny.render(label, True, (255, 255, 255))
        self.screen.blit(text, text.get_rect(center=(x, y)))

    def _drones(self):
        for i, drone in enumerate(self.env.drones):
            p, d = drone.position, drone.destination
            x, y = p.x * self.cell + self.cell // 2, p.y * self.cell + self.cell // 2
            tx, ty = d.x * self.cell + self.cell // 2, d.y * self.cell + self.cell // 2
            col = self.colors[i % len(self.colors)]

            # Trail & Destination line
            self.trails[i].append((x, y))
            self.trails[i] = self.trails[i][-50:]
            if len(self.trails[i]) > 1:
                pygame.draw.lines(self.screen, col, False, self.trails[i], 2)
            pygame.draw.circle(self.screen, col, (tx, ty), max(9, self.cell // 3), 2)

            # Drone body
            radius = max(9, self.cell // 3)
            pygame.draw.circle(self.screen, col, (x, y), radius)
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), radius, 2)
            text = self.tiny.render(str(i + 1), True, (255, 255, 255))
            self.screen.blit(text, text.get_rect(center=(x, y)))

    def _sidebar(self):
        x, y = self.w + 18, 15
        content_width = self.sidebar_w - 36

        self.screen.blit(self.font.render("DRONE TRAFFIC LAB", True, (241, 245, 249)), (x, y))
        y += 31
        mode = f"POLICY: {self.policy_type}"
        self.screen.blit(self.small.render(mode, True, (129, 194, 255)), (x, y))
        y += 21
        self.screen.blit(
            self.tiny.render("CITY: B buildings   X no-fly   W warehouse", True, (180, 193, 212)),
            (x, y),
        )
        self.screen.blit(
            self.tiny.render("D delivery point   C charging station", True, (180, 193, 212)),
            (x, y + 16),
        )
        y += 34
        stats = [
            f"Step: {self.env.step_count} / {self.env.config['max_steps']}",
            f"Deliveries complete: {self.env.total_deliveries} / {self.env.num_drones}",
            f"Route conflicts prevented: {self.env.total_collision_events}",
        ]
        for stat in stats:
            self.screen.blit(self.small.render(stat, True, (203, 213, 225)), (x, y))
            y += 21

        y += 4
        self._separator(x, y, content_width)
        y += 8
        self.screen.blit(self.small.render("MISSIONS  |  press 1–9 or TAB to inspect a drone", True, (203, 213, 225)), (x, y))
        y += 22
        for i, drone in enumerate(self.env.drones):
            status = "DELIVERED" if drone.delivered else ("FLYING" if drone.active else "INACTIVE")
            col = self.colors[i % len(self.colors)]
            if i == self.selected_drone:
                pygame.draw.rect(self.screen, (43, 52, 72), (x - 4, y - 1, content_width, 20), border_radius=3)
            pygame.draw.circle(self.screen, col, (x + 6, y + 8), 5)
            info_text = (
                f"D{i + 1}: ({drone.position.x},{drone.position.y})"
                f" → ({drone.destination.x},{drone.destination.y})"
                f"   {drone.battery:.0f}%   {status}"
            )
            self.screen.blit(self.tiny.render(info_text, True, (226, 232, 240)), (x + 18, y))
            y += 22

        y += 1
        self._separator(x, y, content_width)
        y += 7
        self.screen.blit(
            self.small.render(f"LOCAL MAP INPUT — DRONE {self.selected_drone + 1}", True, (203, 213, 225)),
            (x, y),
        )
        y += 23
        self._draw_local_map(x, y)
        map_size = 2 * self.env.config["observation_radius"] + 1
        y += map_size * 14 + 8
        self._draw_legend(x, y)
        y += 3 * 17 + 4

        self._separator(x, y, content_width)
        y += 7
        self.screen.blit(self.small.render("NEXT ACTION: PROBABILITY", True, (203, 213, 225)), (x, y))
        y += 22
        if self.actions is not None:
            action_names = ("Up", "Down", "Left", "Right", "Wait", "Recharge")
            chosen = int(self.actions[self.selected_drone])
            self.screen.blit(
                self.tiny.render(f"Selected action: {action_names[chosen]}", True, (68, 220, 150)),
                (x, y),
            )
            y += 20
        self._draw_action_probabilities(x, y, content_width)

        footer_y = self.h - 57
        self._separator(x, footer_y - 7, content_width)
        self.screen.blit(self.tiny.render("SPACE pause/resume   N single-step   +/- speed", True, (180, 193, 212)), (x, footer_y))
        self.screen.blit(self.tiny.render("ESC exit   |   Green bar = selected action", True, (180, 193, 212)), (x, footer_y + 20))

        if self.result_summary is not None:
            overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            overlay.fill((8, 12, 20, 190))
            self.screen.blit(overlay, (0, 0))
            title = self.font.render("DELIVERY RUN COMPLETE", True, (255, 255, 255))
            details = self.small.render(self.result_summary, True, (223, 232, 246))
            hint = self.small.render("Press ENTER or ESC to close", True, (129, 194, 255))
            self.screen.blit(title, title.get_rect(center=(self.w // 2, self.h // 2 - 32)))
            self.screen.blit(details, details.get_rect(center=(self.w // 2, self.h // 2 + 2)))
            self.screen.blit(hint, hint.get_rect(center=(self.w // 2, self.h // 2 + 34)))

    def _separator(self, x, y, width):
        pygame.draw.line(self.screen, (51, 65, 85), (x, y), (x + width, y), 1)

    def _draw_local_map(self, x, y):
        maps = self.env.get_local_maps()
        local_map = maps[self.selected_drone]
        tile = 14
        colors = {
            0: (76, 84, 102),
            1: (174, 61, 72),
            2: (234, 179, 8),
            3: (239, 125, 52),
            4: (59, 130, 246),
            5: (34, 197, 94),
        }
        for row in range(local_map.shape[1]):
            for col in range(local_map.shape[2]):
                color = (39, 45, 59)
                for channel, channel_color in colors.items():
                    if local_map[channel, row, col]:
                        color = channel_color
                pygame.draw.rect(
                    self.screen,
                    color,
                    (x + col * tile, y + row * tile, tile - 1, tile - 1),
                )
        center = local_map.shape[1] // 2
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            (x + center * tile, y + center * tile, tile - 1, tile - 1),
            1,
        )

    def _draw_legend(self, x, y):
        labels = (
            ((76, 84, 102), "Buildings"),
            ((174, 61, 72), "No-fly"),
            ((234, 179, 8), "Charger"),
            ((239, 125, 52), "Other drone"),
            ((59, 130, 246), "This drone"),
            ((34, 197, 94), "Destination"),
        )
        for i, (color, label) in enumerate(labels):
            col, row = i % 2, i // 2
            item_x, item_y = x + col * 185, y + row * 17
            pygame.draw.rect(self.screen, color, (item_x, item_y + 2, 9, 9))
            self.screen.blit(
                self.tiny.render(label, True, (203, 213, 225)),
                (item_x + 14, item_y),
            )

    def _draw_action_probabilities(self, x, y, width):
        names = ("Up", "Down", "Left", "Right", "Wait", "Recharge")
        selected = self.selected_drone
        probabilities = None
        if self.action_probabilities is not None:
            probabilities = self.action_probabilities[selected]
        chosen_action = None if self.actions is None else int(self.actions[selected])

        for action, name in enumerate(names):
            row_y = y + action * 25
            probability = 0.0 if probabilities is None else float(probabilities[action])
            label = f"{name:8} {probability:5.1%}"
            color = (34, 197, 94) if action == chosen_action else (96, 110, 134)
            self.screen.blit(self.tiny.render(label, True, (226, 232, 240)), (x, row_y))
            bar_x = x + 91
            bar_y = row_y + 3
            bar_width = max(10, width - 115)
            pygame.draw.rect(self.screen, (47, 55, 72), (bar_x, bar_y, bar_width, 11), border_radius=3)
            if probability > 0:
                pygame.draw.rect(
                    self.screen,
                    color,
                    (bar_x, bar_y, max(2, int(bar_width * probability)), 11),
                    border_radius=3,
                )

    def show_results(self, deliveries, total, conflicts, steps):
        self.result_summary = (
            f"Deliveries {deliveries}/{total}  |  conflicts {conflicts}  |  steps {steps}"
        )
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN
                    and event.key in (pygame.K_RETURN, pygame.K_ESCAPE)
                ):
                    self.close()
                    break

            if not self.running:
                break
            self.screen.fill((18, 20, 26))
            pygame.draw.rect(self.screen, (28, 32, 44), (self.w, 0, self.sidebar_w, self.h))
            self._grid()
            self._city()
            self._drones()
            self._sidebar()
            pygame.display.flip()
            self.clock.tick(30)

    def reset(self):
        self.trails = {i: [] for i in range(self.env.num_drones)}

    def close(self):
        self.running = False
        pygame.quit()