// Container 1
$(function() {
    var content = "My New Awesome Content";
    $(".panel-button").on("click", function() {
        var panelId = $(this).attr("data-panelid");
        $("#" + panelId).slideToggle(200);
        $("#" + panelId + " .panel-body").html(content);
    });

});