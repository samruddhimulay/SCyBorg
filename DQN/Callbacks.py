from stable_baselines3.common.callbacks import BaseCallback


class InfoLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            # -------- Per-Step Logging --------
            self._log_if_present(info, "step_reward", "step/total_reward")
            self._log_if_present(info, "base_reward", "step/base_reward")
            self._log_if_present(info, "action_taken_id", "step/action_taken")
            self._log_if_present(info, "attacks_this_step", "step/attacks_occurred")
            self._log_if_present(info, "server_compromised", "step/server_compromised")
            self._log_if_present(info, "service1_down", "step/service1_down")
            self._log_if_present(info, "service2_down", "step/service2_down")
            self._log_if_present(info, "attack_detected", "step/attack_detected")
            self._log_if_present(info, "security_score", "step/security_score")
            self._log_if_present(info, "active_breaches", "step/active_breaches")
            self._log_if_present(info, "action_successful", "step/action_successful")
            
            # Current episode tracking
            self._log_if_present(info, "episode_successful_defenses", "step/successful_defenses")
            self._log_if_present(info, "episode_unnecessary_actions", "step/unnecessary_actions")
            self._log_if_present(info, "episode_attacks_occurred", "step/attacks_in_episode")
            self._log_if_present(info, "current_episode_reward", "step/cumulative_episode_reward")

            # Log action distribution during episode
            if "action_counts" in info:
                action_names = ['nop', 'patch', 'restore', 'monitor', 'scan']
                for action_id, count in info["action_counts"].items():
                    action_name = action_names[action_id]
                    self.logger.record(f"step/action_count/{action_name}", count)

            # -------- Per-Episode Logging --------
            if "episode_summary" in info:
                episode_info = info["episode_summary"]
                self._log_if_present(episode_info, "total_reward", "episode/total_reward")
                self._log_if_present(episode_info, "episode_length", "episode/length")
                self._log_if_present(episode_info, "average_reward", "episode/average_reward")
                self._log_if_present(episode_info, "security_breaches", "episode/security_breaches")
                self._log_if_present(episode_info, "successful_defenses", "episode/successful_defenses")
                self._log_if_present(episode_info, "unnecessary_actions", "episode/unnecessary_actions")
                self._log_if_present(episode_info, "attacks_occurred", "episode/attacks_occurred")
                self._log_if_present(episode_info, "defense_efficiency", "episode/defense_efficiency")
                self._log_if_present(episode_info, "positive_reward_steps", "episode/positive_reward_steps")
                self._log_if_present(episode_info, "negative_reward_steps", "episode/negative_reward_steps")
                self._log_if_present(episode_info, "average_positive_reward", "episode/average_positive_reward")
                self._log_if_present(episode_info, "average_negative_reward", "episode/average_negative_reward")

                # Log action distribution for completed episode
                if "action_distribution" in episode_info:
                    for action_name, frequency in episode_info["action_distribution"].items():
                        self.logger.record(f"episode/action_distribution/{action_name}", frequency)

        return True

    def _log_if_present(self, info_dict, key, tb_key):
        if key in info_dict:
            self.logger.record(tb_key, info_dict[key])
