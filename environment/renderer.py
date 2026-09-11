import pygame


class Renderer:
    def __init__(self, env, cell_size=40, sidebar_width=300, fps=10):
        self.env, self.cell, self.sidebar_w, self.fps, self.running = env, cell_size, sidebar_width, fps, True
        pygame.init()

        self.w, self.h = env.city.width * cell_size, env.city.height * cell_size
        self.screen = pygame.display.set_mode((self.w + sidebar_width, self.h))
        pygame.display.set_caption("Autonomous Drone Traffic Optimization")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Segoe UI", 16, bold=True)
        self.small = pygame.font.SysFont("Segoe UI", 13)
        self.tiny = pygame.font.SysFont("Segoe UI", 11)

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

    def render(self):
        if not self.running:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        # Backgrounds
        self.screen.fill((18, 20, 26))
        pygame.draw.rect(self.screen, (28, 32, 44), (self.w, 0, self.sidebar_w, self.h))

        self._grid()
        self._city()
        self._drones()
        self._sidebar()

        pygame.display.flip()
        self.clock.tick(self.fps)

    def _grid(self):
        for x in range(self.env.city.width + 1):
            pygame.draw.line(self.screen, (28, 32, 44), (x * self.cell, 0), (x * self.cell, self.h))
        for y in range(self.env.city.height + 1):
            pygame.draw.line(self.screen, (28, 32, 44), (0, y * self.cell), (self.w, y * self.cell))

    def _city(self):
        city = self.env.city
        for r in city.buildings + city.no_fly_zones:
            pygame.draw.rect(
                self.screen, (215, 65, 65, 180),
                (r.x1 * self.cell + 2, r.y1 * self.cell + 2, (r.x2 - r.x1 + 1) * self.cell - 4, (r.y2 - r.y1 + 1) * self.cell - 4),
                border_radius=4
            )

        for p, col, lbl in [(p, (59, 130, 246), "W") for p in city.warehouses] + \
                            [(p, (34, 197, 94), "D") for p in city.delivery_points] + \
                            [(p, (234, 179, 8), "C") for p in city.charging_stations]:
            self._point(p, col, lbl)

    def _point(self, p, color, label):
        x, y = p.x * self.cell + self.cell // 2, p.y * self.cell + self.cell // 2
        pygame.draw.circle(self.screen, color, (x, y), 12)
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
            pygame.draw.line(self.screen, (*col, 100), (x, y), (tx, ty), 1)

            # Drone body
            pygame.draw.circle(self.screen, col, (x, y), 13)
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 13, 2)
            text = self.tiny.render(str(i + 1), True, (255, 255, 255))
            self.screen.blit(text, text.get_rect(center=(x, y)))

    def _sidebar(self):
        x, y = self.w + 20, 20
        
        # Title
        self.screen.blit(self.font.render("TRAFFIC MONITOR", True, (241, 245, 249)), (x, y))
        y += 35

        # Global stats card
        stats = [
            f"Step: {self.env.step_count} / {self.env.config['max_steps']}",
            f"Deliveries: {self.env.total_deliveries} / {self.env.num_drones}",
            f"Collisions: {sum(d.collisions for d in self.env.drones)}",
        ]
        for stat in stats:
            self.screen.blit(self.small.render(stat, True, (148, 163, 184)), (x, y))
            y += 22

        y += 10
        pygame.draw.line(self.screen, (51, 65, 85), (x, y), (x + self.sidebar_w - 40, y), 1)
        y += 15

        # Drones compact list
        for i, drone in enumerate(self.env.drones):
            status = "DELIVERED" if drone.delivered else ("FLYING" if drone.active else "INACTIVE")
            col = self.colors[i % len(self.colors)]
            
            # Indicator dot
            pygame.draw.circle(self.screen, col, (x + 6, y + 7), 5)
            info_text = f"D{i+1}: Bat {drone.battery:3.0f}% | {status}"
            self.screen.blit(self.small.render(info_text, True, (226, 232, 240)), (x + 18, y))
            y += 22

        # Footer controls
        footer_y = self.h - 40
        pygame.draw.line(self.screen, (51, 65, 85), (x, footer_y - 10), (x + self.sidebar_w - 40, footer_y - 10), 1)
        self.screen.blit(self.tiny.render("ESC to Exit | Real-time RL Simulation", True, (100, 116, 139)), (x, footer_y))

    def reset(self):
        self.trails = {i: [] for i in range(self.env.num_drones)}

    def close(self):
        self.running = False
        pygame.quit()