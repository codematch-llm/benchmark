import re

MAX_WORD_NUM = 0
MIN_WORD_NUM = 10

# Supported languages - 23 languages
SUPPORTED_LANGUAGES = [
    'python', 'c', 'cpp', 'php', 'sql', 'ruby', 'javascript', 'java', 'c', 'swift',
    'typescript', 'kotlin', 'scala', 'go', 'rust', 'dart', 'groovy',
    'bash', 'perl', 'r', 'lua', 'haskell', 'clojure'
]

# Define regex for class and function detection for various languages
LANGUAGE_PATTERNS = {
    # codebert supported languages: Python, Java, JavaScript, PHP, Ruby, Go
    'python': {
        'class': r'class\s+(\w+)\s*:',
        'function': r'def\s+(\w+)\s*\(.*?\)\s*:',
    },
    'javascript': {
        'class': r'class\s+(\w+)\s*{',
        'function': r'function\s+(\w+)\s*\(',
    },
    'java': {
        'class': r'(?:public|private|protected)?\s*class\s+(\w+)\s*{',
        'function': r'(public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)?(\w+)\s*\(.*?\)\s*{',
    },
    'php': {
        'class': r'(?:abstract|final)?\s*class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w, ]+)?\s*{',
        'function': r'function\s+(\w+)\s*\(',
    },
    'go': {
        'class': r'type\s+(\w+)\s+struct\s*{',
        'function': r'func\s+\(?(\w+)?\)?\s*\(.*?\)\s*{',
    },
    'ruby': {
        'module': r'module\s+(\w+)\s*',
        'class': r'class\s+(\w+)\s*',
        'function': r'def\s+(\w+)\s*',
    },
    # From down here i didnt check
    'bash': {
        'class': None,  # Bash does not support classes
        'function': r'function\s+(\w+)\s*\(|(\w+)\s*\(\)',
    },
    'lua': {
        'class': None,  # Lua does not have native classes, but we can split by module definitions if needed
        'function': r'function\s+(\w+)\s*\(',
    },
    'swift': {
        'class': r'class\s+(\w+)\s*{',
        'function': r'func\s+(\w+)\s*\(',
    },
    'kotlin': {
        'class': r'class\s+(\w+)\s*{',
        'function': r'fun\s+(\w+)\s*\(',
    },
    'scala': {
        'class': r'class\s+(\w+)\s*{',
        'function': r'def\s+(\w+)\s*\(',
    },
    'lech': {  # Assuming "lech" is a typo; using `c` or `cpp` style as a placeholder
        'class': r'class\s+(\w+)\s*{',
        'function': r'(\w+)\s+(\w+)\s*\(',
    },
    'rust': {
        'class': r'struct\s+(\w+)\s*{',
        'function': r'fn\s+(\w+)\s*\(',
    },
    'dart': {
        'class': r'class\s+(\w+)\s*{',
        'function': r'(\w+)\s+(\w+)\s*\(',
    },
    'groovy': {
        'class': r'class\s+(\w+)\s*{',
        'function': r'def\s+(\w+)\s*\(',
    },
    'perl': {
        'class': None,  # Perl does not use classes traditionally
        'function': r'sub\s+(\w+)\s*{',
    },
    'r': {
        'class': None,  # R does not use classes traditionally
        'function': r'(\w+)\s*<- function\s*\(',
    },
    'sql': {
        'class': None,  # SQL doesn't have classes, but we can match functions or procedures
        'function': r'CREATE\s+(PROCEDURE|FUNCTION)\s+(\w+)\s*\(',
    },
    'haskell': {
        'class': None,  # Haskell doesn't use traditional classes, but we can match data declarations or functions
        'function': r'(\w+)\s*::\s*.*?\n(\w+)\s*=\s*',
    },
    'cpp': {
        'class': r'class\s+(\w+)\s*{',
        'function': r'(\w+)\s+(\w+)\s*\(',
    },
    'clojure': {
        'class': None,  # Clojure doesn't use classes, but we can match function definitions
        'function': r'\(defn\s+(\w+)\s*\[',
    },
}

# Helper function to extract function or class body until the closing brace or end of block
def extract_body(code, start_pos):
    open_braces = 0
    in_function = False
    end_pos = start_pos
    
    while end_pos < len(code):
        char = code[end_pos]
        
        if char == '{':
            open_braces += 1
            in_function = True
        elif char == '}':
            open_braces -= 1
        
        # If we find a balanced set of braces, exit the loop
        if in_function and open_braces == 0:
            break
        end_pos += 1
    
    return code[start_pos:end_pos + 1].strip()  # Return until the closing brace

# Helper function to split code based on regex patterns
def split_code_by_patterns(code, language):
    patterns = LANGUAGE_PATTERNS.get(language, {})
    module_pattern = patterns.get('module')
    class_pattern = patterns.get('class')
    function_pattern = patterns.get('function')
    
    # Initialize dictionary for storing code sections
    code_sections = {}
    global_code = ""

    # Step 1: Attempt to split by modules (Ruby-specific)
    if language == 'ruby' and module_pattern and re.search(module_pattern, code):
        module_matches = list(re.finditer(module_pattern, code))
        
        if module_matches:
            global_code = code[:module_matches[0].start()].strip()  # Capture any code before the first module

            for i, match in enumerate(module_matches):
                module_name = match.group(1)  # Capture the module name
                module_start = match.start()

                # Find the end of the module or the start of the next one
                if i < len(module_matches) - 1:
                    module_body = code[module_start:module_matches[i + 1].start()].strip()
                else:
                    module_body = code[module_start:].strip()

                # Attach global code to the first module
                if i == 0 and global_code:
                    module_body = global_code + "\n" + module_body

                code_sections[f'module_{module_name}'] = module_body

    # Step 2: Attempt to split by classes
    elif class_pattern and re.search(class_pattern, code):
        class_matches = list(re.finditer(class_pattern, code))
        
        if class_matches:
            global_code = code[:class_matches[0].start()].strip() if not code_sections else ""

            for i, match in enumerate(class_matches):
                class_name = match.group(1)  # Capture the class name
                class_start = match.start()

                # Find the end of the class or the start of the next one
                if i < len(class_matches) - 1:
                    class_body = code[class_start:class_matches[i + 1].start()].strip()
                else:
                    class_body = code[class_start:].strip()

                # Attach global code to the first class
                if i == 0 and global_code:
                    class_body = global_code + "\n" + class_body

                code_sections[f'class_{class_name}'] = class_body

    # Step 3: Attempt to split by functions
    elif function_pattern and re.search(function_pattern, code):
        function_matches = list(re.finditer(function_pattern, code))
        
        if function_matches:
            global_code = code[:function_matches[0].start()].strip() if not code_sections else ""

            for i, match in enumerate(function_matches):
                function_name = match.group(1)  # Capture the function name
                function_start = match.start()

                # Find the end of the function or the start of the next one
                if i < len(function_matches) - 1:
                    function_body = code[function_start:function_matches[i + 1].start()].strip()
                else:
                    function_body = code[function_start:].strip()

                # Attach global code to the first function
                if i == 0 and global_code and not code_sections:
                    function_body = global_code + "\n" + function_body

                code_sections[f'func_{function_name}'] = function_body

    # Step 4: If no modules, classes or functions, split by word count
    elif not code_sections:
        return split_into_parts(code)

    return code_sections

# Fallback function for splitting into parts based on word count
def split_into_parts(code, part_size=100):
    words = code.split()
    parts = []
    for i in range(0, len(words), part_size):
        part = " ".join(words[i:i + part_size])
        parts.append(part)
    return {f'part_{i+1}': part for i, part in enumerate(parts)}

# Main function to process code snippets
def preprocess_code(code, language):
    # Step 1: Attempt to split by classes or functions
    code_sections = split_code_by_patterns(code, language)
    
    # Step 2: If no classes or functions, split by word count
    if not code_sections:
        code_sections = split_into_parts(code)
    
    return code_sections

def languages_tests():
    # Python Example usage
    python_code_snippet = """
    global_a = 1
    global_b = 2

    # This is the first function
    def foo():
        # This function prints a message
        print("Hello, World!")

    # Second function
    def bar(x, y):
        # This function adds two numbers and returns the result
        return x + y

    # Third function
    def baz():
        # This function creates a list of numbers and prints them
        numbers = [1, 2, 3, 4, 5]
        for num in numbers:
            print(f"Number: {num}")
    """

    # Preprocess Python code
    print("\n--- python Example ---\n")
    processed_code = preprocess_code(python_code_snippet, 'python')
    for key, val in processed_code.items():
        print(f"Name: {key}\nCode:\n{val}\n-----------")


    # PHP example usage
    php_code_snippet = r'''
    <?php

    namespace App\Models;

    // use Illuminate\Contracts\Auth\MustVerifyEmail;
    use Illuminate\Database\Eloquent\Factories\HasFactory;
    use Illuminate\Foundation\Auth\User as Authenticatable;
    use Illuminate\Notifications\Notifiable;

    class User extends Authenticatable
    {
        /** @use HasFactory<\Database\Factories\UserFactory> */
        use HasFactory, Notifiable;

        /**
        * The attributes that are mass assignable.
        *
        * @var array<int, string>
        */
        protected $fillable = [
            'name',
            'email',
            'password',
        ];

        /**
        * The attributes that should be hidden for serialization.
        *
        * @var array<int, string>
        */
        protected $hidden = [
            'password',
            'remember_token',
        ];

        /**
        * Get the attributes that should be cast.
        *
        * @return array<string, string>
        */
        protected function casts(): array
        {
            return [
                'email_verified_at' => 'datetime',
                'password' => 'hashed',
            ];
        }
    }
    '''

    # Preprocess PHP code
    print("\n--- PHP Example ---\n")
    processed_code = preprocess_code(php_code_snippet, 'php')
    for key, val in processed_code.items():
        print(f"Name: {key}\nCode:\n{val}\n-----------")
        
        
    # JavaScript Example usage
    js_code_snippet = """
    const projectsArea = document.querySelector(".project-list-area");

    function getImagePath(id) {
    return `public/assets/${id}.png`;
    }

    function getProjectUrl(name) {
    return `./projects/${name.toLowerCase().replace(/ /g, '-')}/`;
    }

    function getGithubUrl(name) {
    const repoName = name.toLowerCase().replace(/ /g, '-');
    return `https://github.com/swapnilsparsh/30DaysOfJavaScript/tree/master/projects/${repoName}`;
    }

    function getProjectImg(project) {
    const imgParent = document.createElement("div");
    imgParent.className = "project-img-cont";

    const imgElm = document.createElement("img");
    imgElm.src = project.image;
    imgElm.setAttribute("alt", project.name);

    imgParent.appendChild(imgElm);
    return imgParent;
    }

    function createAnchorElm(href, value, className, attrName, attrValue) {
    const anchorElm = document.createElement("a");
    anchorElm.innerText = value;
    anchorElm.href = href;
    anchorElm.className = className;
    anchorElm.setAttribute(attrName, attrValue);

    return anchorElm;
    }

    function getProjectLinks(project) {
    const linkContainer = document.createElement("div");
    linkContainer.className = "links";

    const websiteLink = createAnchorElm(project.url, 'Live', "btn", "target", "_blank");
    const githubLink = createAnchorElm(project.github, 'Github', "btn", "target", "_blank");

    linkContainer.append(websiteLink, githubLink);
    return linkContainer;
    }

    function getProjectContent(project) {
    const contentContainer = document.createElement("div");
    contentContainer.className = "project-detail";
    const contentElm = document.createElement("div");
    contentElm.className = "project-content";

    const projectName = document.createElement("h2");
    projectName.innerText = project.name;

    const projectDescription = document.createElement("p");
    projectDescription.innerText = project.description;

    contentElm.append(projectName, projectDescription);
    contentContainer.appendChild(contentElm);

    return contentContainer;
    }

    function renderProjectList(projects) {
    projects.forEach((project) => {
        const projectCard = document.createElement("div");
        projectCard.className = "project-card";

        const projectImg = getProjectImg({
        ...project,
        image: getImagePath(project.id),
        });
        const projectContent = getProjectContent(project);
        const projectLinks = getProjectLinks({
        ...project,
        url: getProjectUrl(project.name),
        github: getGithubUrl(project.name),
        });

        projectCard.append(projectImg, projectContent, projectLinks);

        projectsArea.appendChild(projectCard);
    });
    }

    function fetchProjects() {
    fetch("./data.json")
        .then((res) => res.json())
        .then((projects) => {
        renderProjectList(projects);
        })
        .catch((err) => {
        console.log("Error fetching project data:", err);
        });
    }

    fetchProjects();
    """

    # Preprocess JavaScript code
    print("\n--- JavaScript Example ---\n")
    processed_code = preprocess_code(js_code_snippet, 'javascript')
    for key, val in processed_code.items():
        print(f"Name: {key}\nCode:\n{val}\n-----------")
        
    # Java Example usage
    java_code_snippet =  '''package com.example.project;

    public class Project {
        private String name;
        private int id;

        public Project(String name, int id) {
            this.name = name;
            this.id = id;
        }

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public int getId() {
            return id;
        }

        public void setId(int id) {
            this.id = id;
        }

        public void displayProjectInfo() {
            System.out.println("Project Name: " + name);
            System.out.println("Project ID: " + id);
        }
    }

    class Task {
        private String title;
        private boolean completed;

        public Task(String title) {
            this.title = title;
            this.completed = false;
        }

        public String getTitle() {
            return title;
        }

        public boolean isCompleted() {
            return completed;
        }

        public void completeTask() {
            this.completed = true;
            System.out.println("Task " + title + " is completed.");
        }
    }

    public class Main {
        public static void main(String[] args) {
            Project project = new Project("Java Project", 101);
            project.displayProjectInfo();

            Task task = new Task("Implement Java Parsing");
            System.out.println("Task Title: " + task.getTitle());
            task.completeTask();
        }
    }
    ''' 

    # Preprocess Java code
    print("\n--- Java Example ---\n")
    processed_code = preprocess_code(java_code_snippet, 'java')
    for key, val in processed_code.items():
        print(f"Name: {key}\nCode:\n{val}\n-----------")
        
    # Go Example usage
    go_code_snippet =  """
    package main

    import (
        "context"
        "encoding/json"
        "fmt"
        "log"
        "math/rand"
        "net/http"
        "os"
        "sync"
        "time"

        redis "github.com/dicedb/go-dice"
        "github.com/gorilla/websocket"
    )

    var (
        dice    *redis.Client
        upgrade = websocket.Upgrader{
            CheckOrigin: func(r *http.Request) bool {
                return true
            },
        }
    )

    type LeaderboardEntry struct {
        PlayerID  string    `json:"player_id"`
        Score     int       `json:"score"`
        Timestamp time.Time `json:"timestamp"`
    }

    func main() {
        time.Sleep(2 * time.Second)

        dhost := "localhost"
        if val := os.Getenv("DICEDB_HOST"); val != "" {
            dhost = val
        }

        dport := "7379"
        if val := os.Getenv("DICEDB_PORT"); val != "" {
            dport = val
        }

        dice = redis.NewClient(&redis.Options{
            Addr:        fmt.Sprintf("%s:%s", dhost, dport),
            DialTimeout: 10 * time.Second,
            MaxRetries:  10,
        })

        go updateScores()
        go watchLeaderboard()

        // Serve static files for the frontend
        http.Handle("/", http.FileServer(http.Dir(".")))
        http.HandleFunc("/ws", handleWebSocket)

        log.Println("leaderboard running on http://localhost:8000, please open it in your favourite browser.")
        log.Fatal(http.ListenAndServe(":8000", nil))
    }

    func updateScores() {
        ctx := context.Background()
        for {
            entry := LeaderboardEntry{
                PlayerID:  fmt.Sprintf("player:%d", rand.Intn(10)),
                Score:     rand.Intn(100),
                Timestamp: time.Now(),
            }
            lentry, _ := json.Marshal(entry)
            dice.JSONSet(ctx, entry.PlayerID, "$", lentry).Err()
        }
    }

    func watchLeaderboard() {
        ctx := context.Background()
        qwatch := dice.QWatch(ctx)
        qwatch.WatchQuery(ctx, `SELECT $key, $value
                                        WHERE $key LIKE 'player:*' AND '$value.score' > 10
                                        ORDER BY $value.score DESC
                                        LIMIT 5;`)
        defer qwatch.Close()

        ch := qwatch.Channel()
        for {
            select {
            case msg := <-ch:
                entries := toEntries(msg.Updates)
                broadcast(entries)
            case <-ctx.Done():
                return
            }
        }
    }

    func toEntries(updates []redis.KV) []LeaderboardEntry {
        var entries []LeaderboardEntry
        for _, update := range updates {
            var entry LeaderboardEntry
            json.Unmarshal([]byte(update.Value.(string)), &entry)
            entries = append(entries, entry)
        }
        return entries
    }

    func broadcast(entries []LeaderboardEntry) {
        cMux.Lock()
        defer cMux.Unlock()

        message, _ := json.Marshal(entries)
        for client := range clients {
            client.WriteMessage(websocket.TextMessage, []byte(message))
        }
    }

    var (
        clients = make(map[*websocket.Conn]bool)
        cMux    = &sync.Mutex{}
    )

    func handleWebSocket(w http.ResponseWriter, r *http.Request) {
        conn, err := upgrade.Upgrade(w, r, nil)
        if err != nil {
            log.Printf("error upgrading to WebSocket: %v", err)
            return
        }
        defer func(conn *websocket.Conn) {
            err := conn.Close()
            if err != nil {
                log.Printf("error closing WebSocket connection: %v", err)
            }
        }(conn)

        cMux.Lock()
        clients[conn] = true
        cMux.Unlock()

        for {
            _, _, err := conn.ReadMessage()
            if err != nil {
                if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
                    log.Printf("error: %v", err)
                }
                break
            }
        }

        cMux.Lock()
        delete(clients, conn)
        cMux.Unlock()
    }
    """

    # Preprocess Go code
    print("\n--- Go Example ---\n")
    processed_code = preprocess_code(go_code_snippet, 'go')
    for key, val in processed_code.items():
        print(f"Name: {key}\nCode:\n{val}\n-----------")
        
    # Ruby Example usage
    ruby_code_snippet =  """
    # frozen_string_literal: true

    require "active_support/testing/strict_warnings"

    $:.unshift File.expand_path("lib", __dir__)

    ENV["TMPDIR"] = File.expand_path("tmp", __dir__)

    require "active_support/core_ext/kernel/reporting"

    # These are the normal settings that will be set up by Railties
    # TODO: Have these tests support other combinations of these values
    silence_warnings do
    Encoding.default_internal = Encoding::UTF_8
    Encoding.default_external = Encoding::UTF_8
    end

    require "active_support/testing/autorun"
    require "active_support/testing/method_call_assertions"
    require "action_controller"
    require "action_view"
    require "action_view/testing/resolvers"
    require "active_support/dependencies"
    require "active_model"

    module ActionViewTestSuiteUtils
    def self.require_helpers(helpers_dirs)
        Array(helpers_dirs).each do |helpers_dir|
        Dir.glob("#{helpers_dir}/**/*_helper.rb") do |helper_file|
            require helper_file
        end
        end
    end
    end

    ActionViewTestSuiteUtils.require_helpers("#{__dir__}/fixtures/helpers")
    ActionViewTestSuiteUtils.require_helpers("#{__dir__}/fixtures/alternate_helpers")

    Thread.abort_on_exception = true

    # Show backtraces for deprecated behavior for quicker cleanup.
    ActionView.deprecator.debug = true

    # Disable available locale checks to avoid warnings running the test suite.
    I18n.enforce_available_locales = false

    ORIGINAL_LOCALES = I18n.available_locales.map(&:to_s).sort

    FIXTURE_LOAD_PATH = File.expand_path("fixtures", __dir__)

    module RenderERBUtils
    def view
        @view ||= begin
        path = ActionView::FileSystemResolver.new(FIXTURE_LOAD_PATH)
        view_paths = ActionView::PathSet.new([path])
        view = ActionView::Base.with_empty_template_cache
        view.with_view_paths(view_paths)
        end
    end

    def render_erb(string)
        @virtual_path = nil

        template = ActionView::Template.new(
        string.strip,
        "test template",
        ActionView::Template.handler_for_extension(:erb),
        format: :html, locals: [])

        view = ActionView::Base.with_empty_template_cache
        template.render(view.empty, {}).strip
    end
    end

    class BasicController
    attr_accessor :request, :response

    def config
        @config ||= ActiveSupport::InheritableOptions.new(ActionController::Base.config).tap do |config|
        # VIEW TODO: View tests should not require a controller
        public_dir = File.expand_path("fixtures/public", __dir__)
        config.assets_dir = public_dir
        config.javascripts_dir = "#{public_dir}/javascripts"
        config.stylesheets_dir = "#{public_dir}/stylesheets"
        config.assets          = ActiveSupport::InheritableOptions.new(prefix: "assets")
        config
        end
    end
    end

    class ActionDispatch::IntegrationTest < ActiveSupport::TestCase
    self.app = ActionDispatch::MiddlewareStack.new do |middleware|
        middleware.use ActionDispatch::ShowExceptions, ActionDispatch::PublicExceptions.new("#{FIXTURE_LOAD_PATH}/public")
        middleware.use ActionDispatch::DebugExceptions
        middleware.use ActionDispatch::Callbacks
        middleware.use ActionDispatch::Cookies
        middleware.use ActionDispatch::Flash
        middleware.use Rack::Head
    end
    end

    ActionView::RoutingUrlFor.include(ActionDispatch::Routing::UrlFor)

    module ActionController
    class Base
        self.view_paths = FIXTURE_LOAD_PATH

        def self.test_routes(&block)
        routes = ActionDispatch::Routing::RouteSet.new
        routes.draw(&block)
        include routes.url_helpers
        routes
        end
    end

    class TestCase
        include ActionDispatch::TestProcess

        def self.with_routes(&block)
        setup do
            @routes = ActionDispatch::Routing::RouteSet.new
            @routes.draw(&block)

            @controller.singleton_class.include @routes.url_helpers if @controller
        end
        end
    end
    end

    module ActionDispatch
    class DebugExceptions
        private
        remove_method :stderr_logger
        # Silence logger
        def stderr_logger
            nil
        end
    end
    end

    class ActiveSupport::TestCase
    if Process.respond_to?(:fork) && !Gem.win_platform?
        parallelize
    else
        parallelize(with: :threads)
    end

    include ActiveSupport::Testing::MethodCallAssertions
    end

    require_relative "../../tools/test_common"
    """

    # Preprocess Ruby code
    print("\n--- Ruby Example ---\n")
    processed_code = preprocess_code(ruby_code_snippet, 'ruby')
    for key, val in processed_code.items():
        print(f"Name: {key}\nCode:\n{val}\n-----------")
        
# languages_tests