import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from minisweagent import get_repo_root
from minisweagent.run.preprocess.discovery_types import GENERAL_SKILL_TIER

logger = logging.getLogger(__name__)


@dataclass
class SkillDescriptor:
    name: str
    description: str
    path: Path
    tier: str = GENERAL_SKILL_TIER
    loaded: bool = False  # runtime state


class SkillRuntime:
    def __init__(self, allowed_skill_tiers: list[str] | None = None):
        skills_dir = get_repo_root() / "skills"
        self.allowed_skill_tiers = (
            [str(tier).strip() for tier in allowed_skill_tiers if str(tier).strip()]
            if allowed_skill_tiers is not None
            else None
        )
        self.skills = self._discover_skills(skills_dir)

    def _extract_yaml_frontmatter(self, markdown: str) -> dict:
        FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
        match = FRONTMATTER_RE.match(markdown)
        if not match:
            raise ValueError("SKILL.md missing YAML frontmatter")

        return yaml.safe_load(match.group(1))

    def _parse_metadata(self, skill_path: Path) -> SkillDescriptor:
        skill_md = skill_path / "SKILL.md"
        content = skill_md.read_text(encoding="utf-8")

        fm = self._extract_yaml_frontmatter(content)

        return SkillDescriptor(
            name=fm["name"],
            description=fm["description"],
            path=skill_path,
            tier=str(fm.get("tier") or GENERAL_SKILL_TIER).strip() or GENERAL_SKILL_TIER,
            loaded=False,
        )

    def _discover_skills(self, skills_root: Path) -> dict:
        if not skills_root.is_dir():
            logger.debug("Skills directory not found: %s", skills_root)
            return {}
        skills = []
        for p in skills_root.iterdir():
            if p.is_dir() and (p / "SKILL.md").exists():
                try:
                    skills.append(self._parse_metadata(p))
                except Exception as e:
                    print(f"Get skills fail: {e}")
        return {s.name: s for s in skills}

    def _is_skill_allowed(self, skill: SkillDescriptor) -> bool:
        allowed_tiers = getattr(self, "allowed_skill_tiers", None)
        if allowed_tiers is None:
            return True
        return skill.tier in allowed_tiers

    def build_system_prompt(self) -> str:
        blocks = ["\n<available_skills>"]

        for _name, s in self.skills.items():
            if not self._is_skill_allowed(s):
                continue
            blocks.append(
                f"""  <skill>
        <name>{s.name}</name>
        <description>{s.description}</description>
    </skill>"""
            )

        blocks.append("</available_skills>")

        blocks.append(
            """
You can use the above skills to optimize related kernels.
If a skill is relevant, respond with:

```skills
{
"action": "use_skill",
"skill": "<skill-name>"
}
```
Otherwise, respond normally.
    """
        )

        return "\n".join(blocks)

    @staticmethod
    def _list_skill_material_subdirs(skill_root: Path) -> list[Path]:
        """Immediate subdirectories of the skill folder (same directory as SKILL.md), resolved."""
        root = skill_root.resolve()
        if not root.is_dir():
            return []
        out: list[Path] = []
        for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if child.is_dir() and not child.name.startswith("."):
                out.append(child.resolve())
        return out

    @staticmethod
    def _format_skill_material_paths(skill_root: Path) -> str:
        """Human-readable block listing material dirs; empty if none."""
        subdirs = SkillRuntime._list_skill_material_subdirs(skill_root)
        if not subdirs:
            return ""
        bullets = "\n".join(f"- `{p}`" for p in subdirs)
        return (
            "\n\n## Skill material paths\n\n"
            "Material for this skill (e.g. docs/, scripts/) is available under these directories:\n\n"
            f"{bullets}\n"
        )

    def load_skill_by_name(self, skill_name: str) -> dict:
        """Load a skill by its registered name."""
        results = {
            "output": "",
            "returncode": 0,
        }
        if skill_name not in self.skills:
            results["output"] = f"The skill {skill_name} is not exist."
            return results

        skill = self.skills[skill_name]
        if not self._is_skill_allowed(skill):
            results["output"] = f"The skill {skill_name} is not available in this run."
            return results
        if skill.loaded:
            return results

        skill_md = skill.path / "SKILL.md"
        content = skill_md.read_text(encoding="utf-8")
        material = self._format_skill_material_paths(skill.path)
        results["output"] = f"\n# Loaded skill: {skill.name}{material}\n{content}"
        skill.loaded = True
        return results

    def load_skill(self, response: dict) -> dict:
        results = {
            "output": "",
            "returncode": 0,
        }
        if response["content"]:
            match = re.search(r"```skills\s*(\{.*?\})\s*```", response["content"], re.DOTALL)
            if not match:
                return results
            try:
                kill_action = json.loads(match.group(1))
                if kill_action["action"] == "use_skill":
                    return self.load_skill_by_name(kill_action["skill"])
            except Exception as e:
                results["output"] = f"No skills. Error: {e}"
        return results


if __name__ == "__main__":
    skills_str = """
You are an agent that can use the above skills.
If a skill is relevant, respond with:

```skills
{
"action": "use_skill",
"skill": "fused_bucketized-optimization"
}
```
Otherwise, respond normally.
    """
    response = {
        "content": skills_str,
    }
    skill_runtime = SkillRuntime()
    prompt = skill_runtime.build_system_prompt()
    print(prompt)
    skill = skill_runtime.load_skill(response)
    print(skill)
