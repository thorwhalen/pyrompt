"""
GitHub integration for sharing and discovering prompt collections.

Enables:
- Publishing collections to GitHub repositories
- Discovering public collections
- Syncing collections with GitHub
"""

from typing import Optional, List
import os
import tempfile
from pathlib import Path
from collections.abc import MutableMapping

try:
    from github import Github, GithubException
    HAVE_GITHUB = True
except ImportError:
    HAVE_GITHUB = False

try:
    import git
    HAVE_GITPYTHON = True
except ImportError:
    HAVE_GITPYTHON = False


class GitHubPromptCollection(MutableMapping):
    """
    Prompt collection backed by GitHub repository.

    Repositories must end with '_pyrompt' suffix for discovery.
    Contains prompts/ and/or templates/ directories.

    Examples:
        >>> # Publishing a collection
        >>> gh = GitHubPromptCollection(
        ...     repo='username/my_prompts_pyrompt',
        ...     token='ghp_...',
        ...     readonly=False
        ... )
        >>> gh['greeting'] = "Hello, {name}!"
        >>> gh.sync()  # Commits and pushes to GitHub
    """

    def __init__(
        self,
        repo: str,
        *,
        token: Optional[str] = None,
        readonly: bool = True,
        branch: str = 'main',
        local_cache: Optional[str] = None,
        collection_type: str = 'prompts'  # 'prompts' or 'templates'
    ):
        """
        Create GitHub-backed collection.

        Args:
            repo: Repository name (user/repo_name_pyrompt)
            token: GitHub token (for write access)
            readonly: Whether collection is read-only
            branch: Branch to use
            local_cache: Local cache directory
            collection_type: 'prompts' or 'templates'

        Raises:
            ImportError: If PyGithub or GitPython not installed
            ValueError: If repo doesn't end with '_pyrompt'
        """
        if not HAVE_GITHUB:
            raise ImportError(
                "PyGithub required. Install with: pip install PyGithub"
            )

        if not HAVE_GITPYTHON:
            raise ImportError(
                "gitpython required. Install with: pip install gitpython"
            )

        if not repo.endswith('_pyrompt'):
            raise ValueError("Repository must end with '_pyrompt'")

        self.repo_name = repo
        self.readonly = readonly
        self.branch = branch
        self.collection_type = collection_type

        # Setup GitHub client
        self.gh = Github(token) if token else Github()

        try:
            self.repo = self.gh.get_repo(repo)
        except GithubException as e:
            raise ValueError(f"Could not access repository {repo}: {e}")

        # Local cache
        self.local_cache = local_cache or self._default_cache_path()
        self._ensure_local_cache()

        # Set up local store
        from pyrompt.stores import mk_prompt_store, mk_template_store
        cache_path = Path(self.local_cache) / collection_type

        if collection_type == 'prompts':
            self._store = mk_prompt_store(str(cache_path))
        else:
            self._store = mk_template_store(str(cache_path))

    def _default_cache_path(self) -> str:
        """Get default cache path for this repo."""
        cache_base = Path.home() / '.cache' / 'pyrompt' / 'github'
        return str(cache_base / self.repo_name.replace('/', '_'))

    def _ensure_local_cache(self):
        """Clone or pull repo to local cache."""
        if not os.path.exists(self.local_cache):
            # Clone repo
            git.Repo.clone_from(
                f"https://github.com/{self.repo_name}",
                self.local_cache,
                branch=self.branch
            )
        else:
            # Pull latest
            repo = git.Repo(self.local_cache)
            try:
                repo.remotes.origin.pull(self.branch)
            except Exception:
                pass  # Ignore pull errors (might have local changes)

    def sync(self, message: str = "Update prompts via pyrompt"):
        """
        Push local changes to GitHub.

        Args:
            message: Commit message

        Raises:
            PermissionError: If collection is readonly
        """
        if self.readonly:
            raise PermissionError("Collection is readonly")

        repo = git.Repo(self.local_cache)

        # Add all changes
        repo.git.add(A=True)

        # Commit if dirty
        if repo.is_dirty() or repo.untracked_files:
            repo.index.commit(message)

            # Push
            try:
                origin = repo.remote('origin')
                origin.push(self.branch)
            except Exception as e:
                raise RuntimeError(f"Failed to push to GitHub: {e}")

    # MutableMapping interface
    def __getitem__(self, key: str) -> str:
        return self._store[key]

    def __setitem__(self, key: str, value: str):
        if self.readonly:
            raise PermissionError("Collection is readonly")
        self._store[key] = value

    def __delitem__(self, key: str):
        if self.readonly:
            raise PermissionError("Collection is readonly")
        del self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __contains__(self, key):
        return key in self._store


def discover_prompt_collections(
    search_term: Optional[str] = None,
    min_stars: int = 0,
    language: str = 'Python',
    max_results: int = 50
) -> List[dict]:
    """
    Discover public *_pyrompt repositories on GitHub.

    Args:
        search_term: Search query (e.g., "python data")
        min_stars: Minimum star count
        language: Programming language filter
        max_results: Maximum number of results

    Returns:
        List of dicts with repo info (name, description, stars, url)

    Example:
        >>> collections = discover_prompt_collections(
        ...     search_term='python',
        ...     min_stars=5
        ... )
        >>> for repo in collections:
        ...     print(f"{repo['name']}: {repo['stars']} stars")
    """
    if not HAVE_GITHUB:
        raise ImportError(
            "PyGithub required. Install with: pip install PyGithub"
        )

    gh = Github()

    # Build search query
    query_parts = []
    if search_term:
        query_parts.append(search_term)
    query_parts.append('_pyrompt in:name')
    if language:
        query_parts.append(f'language:{language}')
    if min_stars:
        query_parts.append(f'stars:>={min_stars}')

    query = ' '.join(query_parts)

    # Search
    repos = gh.search_repositories(query, sort='stars', order='desc')

    # Format results
    results = []
    for i, repo in enumerate(repos):
        if i >= max_results:
            break

        results.append({
            'name': repo.full_name,
            'description': repo.description,
            'stars': repo.stargazers_count,
            'forks': repo.forks_count,
            'url': repo.html_url,
            'updated': repo.updated_at.isoformat() if repo.updated_at else None
        })

    return results


def fork_collection(
    source: str,
    token: str,
    organization: Optional[str] = None
) -> 'GitHubPromptCollection':
    """
    Fork a collection to your account or organization.

    Args:
        source: Source repo (user/repo_pyrompt)
        token: GitHub token with repo permissions
        organization: Optional organization to fork to

    Returns:
        GitHubPromptCollection for the new fork

    Example:
        >>> forked = fork_collection(
        ...     'thorwhalen/awesome_prompts_pyrompt',
        ...     token='ghp_...'
        ... )
    """
    if not HAVE_GITHUB:
        raise ImportError(
            "PyGithub required. Install with: pip install PyGithub"
        )

    gh = Github(token)
    source_repo = gh.get_repo(source)

    # Create fork
    if organization:
        fork = source_repo.create_fork(organization=organization)
    else:
        fork = source_repo.create_fork()

    # Return collection for fork
    return GitHubPromptCollection(
        repo=fork.full_name,
        token=token,
        readonly=False
    )


def clone_collection(
    repo: str,
    local_path: str,
    token: Optional[str] = None
):
    """
    Clone a GitHub collection to local directory.

    Args:
        repo: Repository name (user/repo_pyrompt)
        local_path: Local directory path
        token: Optional GitHub token

    Example:
        >>> clone_collection(
        ...     'thorwhalen/awesome_prompts_pyrompt',
        ...     '/tmp/my_prompts'
        ... )
    """
    if not HAVE_GITPYTHON:
        raise ImportError(
            "gitpython required. Install with: pip install gitpython"
        )

    url = f"https://github.com/{repo}"
    if token:
        # Construct authenticated URL
        url = f"https://{token}@github.com/{repo}"

    git.Repo.clone_from(url, local_path)
